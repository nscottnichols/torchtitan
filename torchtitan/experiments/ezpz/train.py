import os
import sys
import datetime

from typing import Any

import ezpz
import ezpz.dist
import ezpz.utils
import torch
import torch.distributed

from torchtitan.config import ConfigManager
from torchtitan.experiments.ezpz.logging import init_logger
from torchtitan.tools.logging import logger

DEFAULT_MODULE = "ezpz.agpt"
DEFAULT_CONFIG = "ezpz_agpt_debugmodel"

_LEGACY_KEY_REMAP = {
    "job.dump-folder": "dump-folder",
    "job.print-config": "debug.print-config",
    "job.print-args": "debug.print-config",
    "job.no-print-config": "debug.no-print-config",
    "job.save-config-file": "debug.save-config-file",
    "model.hf-assets-path": "hf-assets-path",
    "model.tokenizer-path": "hf-assets-path",
    "training.dataset": "dataloader.dataset",
    "training.dataset-path": "dataloader.dataset-path",
    "validation.enable": "validator.enable",
    "validation.no-enable": "validator.no-enable",
    "validation.freq": "validator.freq",
    "validation.steps": "validator.steps",
    "validation.dataset": "validator.dataloader.dataset",
    "validation.dataset-path": "validator.dataloader.dataset-path",
}

_FLAVOR_TO_CONFIG = {
    "debug": "ezpz_agpt_debugmodel",
    "debugmodel": "ezpz_agpt_debugmodel",
    "2b": "ezpz_agpt_2b",
    "7b": "ezpz_agpt_7b",
    "8b": "ezpz_agpt_8b",
    "auroragpt-2b": "ezpz_agpt_2b",
    "auroragpt2b": "ezpz_agpt_2b",
    "auroragpt-7b": "ezpz_agpt_7b",
    "auroragpt7b": "ezpz_agpt_7b",
    "llama3-8b": "ezpz_agpt_8b",
}


def _update_env() -> None:
    os.environ.setdefault("WANDB_PROJECT", "torchtitan.ezpz.train")
    now = datetime.datetime.now()
    dstr = now.strftime("%Y-%m-%d-%H%M%S")
    env_dict = {
        f"env.{k}": v
        for k, v in dict(os.environ).items()
        if not k.startswith("_") and "API" not in k
    }
    env_dict |= {
        "created_at": dstr,
        "day": ezpz.utils.get_timestamp("%d"),
        "ezpz_file": ezpz.__file__,
        "ezpz_version": getattr(ezpz, "__version__", None),
        "hostname": ezpz.dist.get_hostname(),
        "month": ezpz.utils.get_timestamp("%m"),
        "machine": ezpz.dist.get_machine(),
        "pytorch_backend": str(ezpz.dist.get_torch_backend()).lower(),
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "world_size": ezpz.dist.get_world_size(),
        "year": ezpz.utils.get_timestamp("%Y"),
        "working_directory": os.getcwd(),
    }
    _ = env_dict.pop("LS_COLORS", None)
    _ = env_dict.pop("PS1", None)
    logger.info(f"Running on {ezpz.dist.get_machine()=}")
    logger.info(f"{env_dict=}")


def _has_flag(args: list[str], name: str) -> bool:
    key = f"--{name}"
    return any(arg == key or arg.startswith(f"{key}=") for arg in args)


def _inject_default_module_and_config(args: list[str]) -> list[str]:
    merged = list(args)
    if not _has_flag(merged, "module"):
        merged = ["--module", DEFAULT_MODULE, *merged]
    if not _has_flag(merged, "config"):
        merged = ["--config", DEFAULT_CONFIG, *merged]
    return merged


def _canonicalize_option(option: str) -> str:
    return option.removeprefix("--").replace("_", "-")


def _config_name_from_flavor(flavor: str) -> str:
    normalized = flavor.strip().lower()
    return _FLAVOR_TO_CONFIG.get(normalized, f"ezpz_agpt_{normalized}")


def _translate_legacy_args(args: list[str]) -> list[str]:
    translated: list[str] = []
    legacy_tokenizer_backend: str | None = None
    i = 0

    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            translated.append(token)
            i += 1
            continue

        inline_value = "=" in token
        if inline_value:
            option, value = token.split("=", 1)
            consume_next = False
        else:
            option = token
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                consume_next = True
            else:
                value = None
                consume_next = False

        key = _canonicalize_option(option)

        if key == "job.config-file":
            raise ValueError(
                "`--job.config-file` is no longer supported for ezpz. "
                "Use `--module ezpz.agpt --config ezpz_agpt_debugmodel` and CLI overrides instead."
            )

        if key in {"experimental.custom-args-module", "experimental.custom-import"}:
            logger.warning("Ignoring deprecated --experimental.* flag for ezpz.")
            i += 2 if consume_next else 1
            continue

        if key == "model.name":
            if value is not None:
                module_name = value
                if value.strip().lower() == "blendcorpus":
                    module_name = DEFAULT_MODULE
                translated.extend(["--module", module_name])
            i += 2 if consume_next else 1
            continue

        if key == "model.flavor":
            if value is not None:
                translated.extend(["--config", _config_name_from_flavor(value)])
            i += 2 if consume_next else 1
            continue

        if key == "model.tokenizer-backend":
            if value is not None:
                legacy_tokenizer_backend = value
            i += 2 if consume_next else 1
            continue

        if key.startswith("blendcorpus."):
            remapped = f"dataloader.{key.removeprefix('blendcorpus.')}"
        else:
            remapped = _LEGACY_KEY_REMAP.get(key, key)

        if value is None:
            translated.append(f"--{remapped}")
        else:
            translated.extend([f"--{remapped}", value])

        i += 2 if consume_next else 1

    if legacy_tokenizer_backend is not None:
        translated.extend(
            ["tokenizer:config", "--tokenizer.backend", legacy_tokenizer_backend]
        )

    return translated


def _ensure_rank_env() -> None:
    os.environ.setdefault("LOCAL_RANK", str(ezpz.dist.get_local_rank()))
    if torch.distributed.is_initialized():
        os.environ.setdefault("RANK", str(torch.distributed.get_rank()))
        os.environ.setdefault("WORLD_SIZE", str(torch.distributed.get_world_size()))


def main(args: list[str] | None = None) -> None:
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    raw_args = sys.argv[1:] if args is None else args
    parsed_args = _inject_default_module_and_config(_translate_legacy_args(raw_args))
    config_manager = ConfigManager()
    config: Any = config_manager.parse_args(parsed_args)
    trainer = None

    try:
        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        trainer = config.build()

        if config.checkpoint.create_seed_checkpoint:
            assert int(os.environ["WORLD_SIZE"]) == 1, (
                "Must create seed checkpoint using a single device, to disable sharding."
            )
            assert config.checkpoint.enable, (
                "Must enable checkpointing when creating a seed checkpoint."
            )
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    ezpz.dist.setup_torch()
    _ensure_rank_env()
    _update_env()
    main()
