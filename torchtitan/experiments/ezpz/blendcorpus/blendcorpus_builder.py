from dataclasses import dataclass
import importlib
import os
from types import SimpleNamespace
from typing import Any

import torch

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.logging import logger


def _import_blendcorpus_modules():
    try:
        bc_mpu = importlib.import_module("blendcorpus.parallel_state")
        bc_config_mod = importlib.import_module("blendcorpus.data.config")
        bc_sampler_mod = importlib.import_module("blendcorpus.data.data_samplers")
        bc_dataset_mod = importlib.import_module("blendcorpus.data.gpt_dataset")
    except ImportError as exc:
        raise ImportError(
            "BlendCorpus dataset was requested but `blendcorpus` is not installed. "
            "Install it first (for example: `pip install blendcorpus`) or set "
            "`--dataloader.dataset` to a non-blendcorpus dataset."
        ) from exc

    bc_get_config = getattr(bc_config_mod, "get_config")
    bc_set_config = getattr(bc_config_mod, "set_config")
    build_pretraining_data_loader = getattr(
        bc_sampler_mod, "build_pretraining_data_loader"
    )
    build_gpt_datasets = getattr(bc_dataset_mod, "build_gpt_datasets")

    return (
        bc_mpu,
        bc_get_config,
        bc_set_config,
        build_gpt_datasets,
        build_pretraining_data_loader,
    )


class BlendCorpusDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        num_workers: int = 0
        persistent_workers: bool = False
        pin_memory: bool = True
        prefetch_factor: int | None = None
        infinite: bool = True

        split: str = "95,5,0"
        dataloader_type: str = "single"
        shuffle: bool = True
        shuffle_sample_in_corpus: bool = True
        blend_sample_in_corpus: bool = False
        append_eod: bool = True
        provide_attention_mask: bool = False
        eod_token_id: int | None = None
        data_cache_path: str = ".cache/blendcorpus"

        train_iters: int | None = None

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        self._mode = "hf"
        self._delegate: BaseDataLoader | None = None

        if config.dataset != "blendcorpus":
            hf_cfg = HuggingFaceTextDataLoader.Config(
                dataset=config.dataset,
                dataset_path=config.dataset_path,
                num_workers=config.num_workers,
                persistent_workers=config.persistent_workers,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                infinite=config.infinite,
            )
            self._delegate = hf_cfg.build(
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                tokenizer=tokenizer,
                seq_len=seq_len,
                local_batch_size=local_batch_size,
            )
            return

        self._mode = "blendcorpus"
        (
            bc_mpu,
            bc_get_config,
            bc_set_config,
            build_gpt_datasets,
            build_pretraining_data_loader,
        ) = _import_blendcorpus_modules()

        parallel_dims = kwargs.get("parallel_dims")
        tp_degree = getattr(parallel_dims, "tp", 1)
        pp_degree = getattr(parallel_dims, "pp", 1)
        cp_degree = getattr(parallel_dims, "cp", 1)

        requested_global_batch_size = kwargs.get("global_batch_size")
        if not requested_global_batch_size or requested_global_batch_size <= 0:
            requested_global_batch_size = local_batch_size * dp_world_size

        train_iters = config.train_iters
        if train_iters is None:
            train_iters = kwargs.get("training_steps")
        if train_iters is None:
            train_iters = 1
            logger.warning(
                "BlendCorpus train_iters was not provided; defaulting to 1. "
                "Set --training.steps or --dataloader.train-iters explicitly."
            )
        else:
            train_iters = int(train_iters)
            if train_iters <= 0:
                logger.warning(
                    "BlendCorpus got non-positive train_iters=%s; defaulting to 1 "
                    "to avoid oversized index allocation.",
                    train_iters,
                )
                train_iters = 1

        bc_cfg = SimpleNamespace(
            data_file_list=config.dataset_path,
            seq_length=seq_len,
            train_iters=train_iters,
            eval_iters=0,
            seed=42,
            data_impl="mmap",
            micro_batch_size=int(local_batch_size),
            global_batch_size=int(requested_global_batch_size),
            tensor_model_parallel_size=int(tp_degree),
            pipeline_model_parallel_size=int(pp_degree),
            sequence_parallel_size=int(cp_degree),
            num_workers=int(config.num_workers),
            split=config.split,
            dataloader_type=config.dataloader_type,
            shuffle=bool(config.shuffle),
            shuffle_sample_in_corpus=bool(config.shuffle_sample_in_corpus),
            blend_sample_in_corpus=bool(config.blend_sample_in_corpus),
            append_eod=bool(config.append_eod),
            provide_attention_mask=bool(config.provide_attention_mask),
            eod_token_id=(
                int(config.eod_token_id)
                if config.eod_token_id is not None
                else getattr(tokenizer, "eos_id", None)
            ),
            data_cache_path=os.path.abspath(config.data_cache_path),
        )
        os.makedirs(bc_cfg.data_cache_path, exist_ok=True)

        bc_mpu.initialize_model_parallel(
            tensor_model_parallel_size=bc_cfg.tensor_model_parallel_size,
            pipeline_model_parallel_size=bc_cfg.pipeline_model_parallel_size,
            sequence_parallel_size=bc_cfg.sequence_parallel_size,
        )

        bc_set_config(bc_cfg)
        self._bc_cfg = bc_get_config()

        # Build indices on rank 0 first, then barrier so other ranks wait
        # for the files to be fully written to the parallel filesystem.
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            build_gpt_datasets(self._bc_cfg)
        torch.distributed.barrier()

        train_ds, _, _ = build_gpt_datasets(self._bc_cfg)
        self._train_ds = train_ds
        self._build_pretraining_data_loader = build_pretraining_data_loader
        self._loader = build_pretraining_data_loader(train_ds, 0, self._bc_cfg)
        self._consumed_samples = 0

        try:
            self._len = len(self._loader)
        except TypeError:
            self._len = int(1e12)

        logger.info("Using BlendCorpus dataloader backend")

    def __len__(self):
        if self._delegate is not None:
            if hasattr(self._delegate, "__len__"):
                return len(self._delegate)  # type: ignore[arg-type]
            return int(1e12)
        return self._len

    def __iter__(self):
        if self._delegate is not None:
            yield from iter(self._delegate)
            return

        for batch in self._loader:
            tokens = batch["text"].long()
            input_ids = tokens[:, :-1].contiguous()
            labels = tokens[:, 1:].contiguous()
            yield {"input": input_ids}, labels

    def set_consumed_by_global_step(self, global_step: int, global_batch_size: int):
        if self._delegate is not None:
            return

        consumed = int(global_step) * int(global_batch_size)
        self._consumed_samples = consumed
        self._loader = self._build_pretraining_data_loader(
            self._train_ds, consumed, self._bc_cfg
        )

    def state_dict(self) -> dict[str, Any]:
        if self._delegate is not None:
            return self._delegate.state_dict()
        return {"consumed_samples": int(self._consumed_samples)}

    def load_state_dict(self, state_dict: dict[str, Any]):
        if self._delegate is not None:
            self._delegate.load_state_dict(state_dict)
            return

        consumed = int(state_dict.get("consumed_samples", 0))
        if consumed != self._consumed_samples:
            self._consumed_samples = consumed
            self._loader = self._build_pretraining_data_loader(
                self._train_ds, consumed, self._bc_cfg
            )
