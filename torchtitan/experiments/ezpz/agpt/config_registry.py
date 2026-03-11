import ezpz
import ezpz.distributed

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig, CommConfig, TrainingConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
    BlendCorpusDataLoader,
)
from torchtitan.experiments.ezpz.blendcorpus.build_tokenizer import EZPZTokenizer
from torchtitan.experiments.ft.config.job_config import FaultTolerance
from torchtitan.experiments.ft.trainer import FaultTolerantTrainer

from . import model_registry


def agpt_debugmodel() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_debugmodel()


def agpt_2b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_2b()


def agpt_2b_hf() -> FaultTolerantTrainer.Config:
    cfg = ezpz_agpt_2b()
    cfg.dataloader.dataset_path = None
    return cfg


def agpt_2b_flex_attn() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_2b_flex_attn()


def agpt_7b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_7b()


def agpt_7b_hf() -> FaultTolerantTrainer.Config:
    cfg = ezpz_agpt_7b()
    cfg.dataloader.dataset_path = None
    return cfg


def ezpz_agpt_8b() -> FaultTolerantTrainer.Config:
    return _base_config("8B")


def agpt_8b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_8b()


def _base_config(flavor: str) -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry(flavor),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10000,
        ),
        dataloader=BlendCorpusDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        comm=CommConfig(train_timeout_seconds=100),
        fault_tolerance=FaultTolerance(enable=False),
        validator=Validator.Config(enable=False),
    )


def ezpz_agpt_debugmodel() -> FaultTolerantTrainer.Config:
    cfg = _base_config("debugmodel")
    cfg.hf_assets_path = "./assets/test"
    cfg.metrics.enable_wandb = True
    cfg.debug.print_config = True
    cfg.training.local_batch_size = 2
    cfg.training.seq_len = 8192
    cfg.training.dtype = "bfloat16"
    cfg.dataloader.dataset = "blendcorpus"
    machine_name = ezpz.distributed.get_machine().lower()
    cfg.dataloader.dataset_path = (
        f"torchtitan/experiments/ezpz/data-lists/{machine_name}/books.txt"
    )
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_wandb = True
    cfg.compile = CompileConfig(enable=True)
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = 50
    return cfg


def ezpz_agpt_2b() -> FaultTolerantTrainer.Config:
    cfg = _base_config("2b")
    cfg.hf_assets_path = "./assets/hf/gemma-7b"
    cfg.debug.print_config = True
    cfg.training.local_batch_size = 2
    cfg.training.seq_len = 8192
    cfg.training.dtype = "bfloat16"
    cfg.dataloader.dataset = "blendcorpus"
    machine_name = ezpz.distributed.get_machine().lower()
    cfg.dataloader.dataset_path = (
        f"torchtitan/experiments/ezpz/data-lists/{machine_name}/books.txt"
    )
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_wandb = True
    cfg.compile = CompileConfig(enable=True)
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = 50
    return cfg


def ezpz_agpt_2b_flex_attn() -> FaultTolerantTrainer.Config:
    cfg = _base_config("2b_flex_attn")
    cfg.hf_assets_path = "./assets/hf/gemma-7b"
    cfg.debug.print_config = True
    cfg.training.local_batch_size = 2
    cfg.training.seq_len = 8192
    cfg.training.dtype = "bfloat16"
    cfg.dataloader.dataset = "blendcorpus"
    machine_name = ezpz.distributed.get_machine().lower()
    cfg.dataloader.dataset_path = (
        f"torchtitan/experiments/ezpz/data-lists/{machine_name}/books.txt"
    )
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_wandb = True
    cfg.compile = CompileConfig(enable=True)
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = 50
    return cfg


def ezpz_agpt_7b() -> FaultTolerantTrainer.Config:
    cfg = _base_config("7b")
    cfg.hf_assets_path = "./assets/hf/llama-2-7b-hf"
    cfg.debug.print_config = True
    cfg.training.local_batch_size = 2
    cfg.training.seq_len = 4096
    cfg.training.dtype = "bfloat16"
    cfg.dataloader.dataset = "blendcorpus"
    machine_name = ezpz.distributed.get_machine().lower()
    cfg.dataloader.dataset_path = (
        f"torchtitan/experiments/ezpz/data-lists/{machine_name}/books.txt"
    )
    cfg.metrics.enable_wandb = True
    cfg.compile = CompileConfig(enable=True)
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = 50
    return cfg


def ezpz_agpt_20b() -> FaultTolerantTrainer.Config:
    cfg = _base_config("20b")
    cfg.hf_assets_path = "./assets/hf/gemma-7b"
    cfg.debug.print_config = True
    cfg.training.local_batch_size = 1
    cfg.training.seq_len = 8192
    cfg.training.dtype = "bfloat16"
    cfg.dataloader.dataset = "blendcorpus"
    machine_name = ezpz.distributed.get_machine().lower()
    cfg.dataloader.dataset_path = (
        f"torchtitan/experiments/ezpz/data-lists/{machine_name}/books.txt"
    )
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_wandb = True
    cfg.compile = CompileConfig(enable=True)
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = 50
    return cfg


def agpt_20b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_20b()


def ezpz_agpt_50b() -> FaultTolerantTrainer.Config:
    cfg = _base_config("50b")
    cfg.hf_assets_path = "./assets/hf/gemma-7b"
    cfg.training.local_batch_size = 1
    cfg.training.seq_len = 8192
    cfg.training.dtype = "bfloat16"
    cfg.dataloader.dataset = "blendcorpus"
    machine_name = ezpz.distributed.get_machine().lower()
    cfg.dataloader.dataset_path = (
        f"torchtitan/experiments/ezpz/data-lists/{machine_name}/books.txt"
    )
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_wandb = True
    cfg.compile = CompileConfig(enable=True)
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = 50
    return cfg


def agpt_50b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_50b()
