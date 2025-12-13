# TorchTitan + BlendCorpus with Pytorch 2.8 on Aurora

> Sam Foreman  
> 2025-12-12

Complete instructions for setting up and running TorchTitan with BlendCorpus
integration on Aurora.

We use a virtual environment built on top of the conda environment provided
from the `frameworks/2025.2.0` module.

## 🆘 Helpers and QOL Improvements

1. The isolated directory used below (`NOW`) is optional (maybe useful)
2. The `uvi` alias is just a convenience (but useful, imo)

```bash
NOW="$(date "+%Y-%m-%d-%H%M%S")" ; mkdir -p "tmp/${NOW}" && cd "tmp/${NOW}"
alias uvi="uv pip install --no-cache --link-mode=copy"
```

## ⚙️ Job Config and HF Repo for Downloading Assets

```bash
rn="meta-llama/llama-2-7b-hf"  # repository name  (for downloading)
hfp="assets/hf/llama-2-7b-hf"  # hf assets path   (for saving)
#                              # job.config_file  (for training)
cfg="torchtitan/experiments/blendcorpus/train_configs/auroraGPT_7B.toml"
```

## 📦 Download Repo and HF Assets

- ⬇️ Clone repository: [saforem2/torchtitan](https://github.com/saforem2/torchtitan)

  ```bash
  git clone "https://github.com/saforem2/torchtitan"
  cd torchtitan
  ```

  - 🌱 Checkout branch: [saforem2/blendcorpus](https://github.com/saforem2/torchtitan/tree/saforem2/blendcorpus)

    ```bash
    git checkout saforem2/blendcorpus
    ```

- 📦 Download HF assets:

  ```bash
  python3 scripts/download_hf_assets.py --repo_id "${rn}" --assets tokenizer
  ```

## 🐍 Setup Python and Install Dependencies


1. Setup Python:

    ```bash
    source <(curl -sL https://bit.ly/ezpz-utils) && ezpz_setup_env
    ```

1. Install dependencies:

   - [saforem2/ezpz](https://github.com/saforem2/ezpz):

      ```bash
      uvi "git+https://github.com/saforem2/ezpz@distributed-metrics"
      ```

   - [zhenghh04/blendcorpus](https://github.com/zhenghh04/blendcorpus):

      ```bash
      mkdir deps && git clone https://github.com/zhenghh04/blendcorpus deps/blendcorpus
      uvi -e deps/blendcorpus --no-deps
      cd deps/blendcorpus/blendcorpus/data && make && cd -
      ```

## 🚀 Launch Training

- Launch:

  ```bash
  ezpz-launch python3 -m torchtitan.experiments.blendcorpus.train \
      --job.config_file "${cfg}" \
      --model.hf_assets_path assets/hf/llama-2-7b-hf
  ```
