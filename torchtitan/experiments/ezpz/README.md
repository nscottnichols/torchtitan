# TorchTitan + 🍋 `ezpz`

> [!NOTE]
> These instructions assume we are using the fork `saforem2/torchtitan`, on
> the branch `ezpz`, i.e.:  
> [saforem2/torchtitan@ezpz](https://github.com/saforem2/torchtitan/tree/ezpz)

## Setup TorchTitan

```bash
qsub -q next-eval -A AuroraGPT -l walltime=06:00:00,filesystems=flare:home -l select=2 -I
git clone https://github.com/saforem2/torchtitan --branch ezpz
cd torchtitan
```

## Install Dependencies

```bash
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
uv pip install "git+https://github.com/saforem2/ezpz@dev"
uv pip install "git+https://github.com/zhenghh04/blendcorpus"
uv pip install tensorboard tyro
```

## Launch Training

```bash
MODEL=2b
MBS=2
SEQ_LEN=8192

ezpz launch python3 -m torchtitan.experiments.ezpz.train \
   --module ezpz.agpt \
   --config "ezpz_agpt_${MODEL}" \
   --training.seq-len "${SEQ_LEN}" \
   --training.local_batch_size "${MBS}" \
   --debug.print-config \
   --metrics.log_freq=1 \
   --metrics.enable_wandb \
   --compile.enable \
   --training.dataset blendcorpus \
   --training.dataset_path torchtitan/experiments/ezpz/data-lists/aurora/books.txt \
   --model.tokenizer_backend hf
```

## References


- 🍋 `ezpz`:
  - Documentation: [ezpz.cool](https://ezpz.cool)
  - GitHub: [saforem2/ezpz](https://github.com/saforem2/ezpz)
