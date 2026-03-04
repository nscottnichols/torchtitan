# TorchTitan + 🍋 `ezpz`

> [!NOTE]
> These instructions assume we are using the fork `saforem2/torchtitan`, on
> the branch `ezpz`, i.e.:  
> [saforem2/torchtitan@ezpz](https://github.com/saforem2/torchtitan/tree/ezpz)

1. Submit job:

   - Aurora:

     ```bash
     # [03/04/2026] `-q next-eval` required for PyTorch 2.10 on Aurora
     qsub -q next-eval -A <project> -l walltime=06:00:00,filesystems=flare:home -l select=2 -I
     ```
   
1. Clone TorchTitan from [saforem2/torchtitan@ezpz](https://github.com/saforem2/torchtitan/blob/ezpz):

   ```bash
   git clone https://github.com/saforem2/torchtitan --branch ezpz
   cd torchtitan
   ```

1. Setup environment:

   ```bash
   source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
   ```

1. Install Dependencies

   ```bash
   uv pip install "git+https://github.com/saforem2/ezpz"
   uv pip install "git+https://github.com/zhenghh04/blendcorpus"
   uv pip install tensorboard tyro
   ```

1. Launch Training

   - AuroraGPT-2B:
   
     ```bash
     MODEL=2b
     DFL=torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt
     ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module ezpz.agpt \
        --config "ezpz_agpt_${MODEL}" \
        --training.dataset_path "${DFL}" \
        --debug.print_config
     ```
   
   - AuroraGPT-7B:
   
     ```bash
     MODEL=7b
     DFL=torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt
     ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module ezpz.agpt \
        --config "ezpz_agpt_${MODEL}" \
        --training.dataset_path "${DFL}" \
        --debug.print_config
     ```


## References

- 🍋 `ezpz`:
  - Documentation: [ezpz.cool](https://ezpz.cool)
  - GitHub: [saforem2/ezpz](https://github.com/saforem2/ezpz)
