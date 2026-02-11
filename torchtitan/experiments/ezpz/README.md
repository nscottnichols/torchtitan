# Experiments: 🍋 `ezpz`

## Setup

> [!NOTE]
> These instructions assume we are using the fork `saforem2/torchtitan`, on
> the branch `ezpz`, i.e.:  
> [saforem2/torchtitan@ezpz](https://github.com/saforem2/torchtitan/tree/ezpz)

1. Clone and navigate into repo:

   ```bash
   git clone --branch ezpz https://github.com/saforem2/torchtitan
   cd torchtitan
   ```

1. Setup Python Environment:

   - <details closed><summary>Manual Python Setup:</summary>
    
     - Already have an environment with `pytorch>=2.10` and `mpi4py`?
    
       ```bash
       uv venv --python=$(which python3) --system-site-packages
       source .venv/bin/activate
       ```
    
       - Otherwise:
    
         ```bash
         uv venv --python=3.12
         source .venv/bin/activate
         ```
    
     </details>
    
    - Automatic environment setup:
    
      ```bash
      source <(curl -fsSL https://bit.ly/ezpz-utils)
      ezpz_setup_env
      ```


1. Install `ezpz`:

   ```bash
   uv pip install "git+https://github.com/saforem2/ezpz"
   ```


1. Run AuroraGPT-2B training example:

   - Download `google/gemma-7b` tokenizer for AuroraGPT-2B:

     ```bash
     python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
     ```

   - Launch training:

     ```bash
     ezpz launch python3 -m torchtitan.experiments.ezpz.train \
          --model.name ezpz.agpt \
          --model.flavor 2b \
          --training.seq_len 8192 \
          --training.local_batch_size=1 \
          --model.hf_assets_path assets/hf/gemma-7b \
          --job.print_config \
          --compile.enable \
          --training.global_batch_size=48
     ```


## References


- 🍋 `ezpz`:
  - Documentation: [ezpz.cool](https://ezpz.cool)
  - GitHub: [saforem2/ezpz](https://github.com/saforem2/ezpz)







