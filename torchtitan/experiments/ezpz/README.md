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
         --job.print_config --compile.enable
         --training.global_batch_size=48 \
         --metrics.log_freq=1
         --metrics.enable_wandb
     ```

     - <details closed><summary>Output from Sunspot:</summary>

       Running on two nodes of Sunspot using the new
       `frameworks_2025.3.1-RC4`[^rc4] with `torch==2.10`:
         
       ```bash
       #[frameworks_2025.3.1-RC4](torchtitan-frameworks_2025.3.1-RC4)
       #[/t/d/f/p/s/torchtitan][ezpz][?] [<U+F051B>  20s]
       #[02/11/26 @ 00:53:00][x1921c1s5b0n0]
       ; ezpz launch python3 -m torchtitan.experiments.ezpz.train --model.name ezpz.agpt --model.flavor 2b --training.seq_len 8192 --training.local_batch_size=1 --model.hf_assets_path assets/hf/gemma-7b --job.print_config --compile.enable --training.global_batch_size=48 --metrics.log_freq=1 --metrics.enable_wandb

         
       [2026-02-11 00:53:07,639158][I][ezpz/launch:421:launch] ----[🍋 ezpz.launch][started][2026-02-11-005307]----
       [2026-02-11 00:53:07,644171][I][ezpz/launch:66:_log_json_log_file] Logs available at: /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/logs/torchtitan.experiments.ezpz.train/2026-02-11-005306-rank0.jsonl
       [2026-02-11 00:53:08,499798][I][ezpz/launch:442:launch] Job ID: 12461964
       [2026-02-11 00:53:08,500562][I][ezpz/launch:443:launch] nodelist: ['x1921c1s5b0n0', 'x1921c1s7b0n0']
       [2026-02-11 00:53:08,500961][I][ezpz/launch:444:launch] hostfile: /var/spool/pbs/aux/12461964.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
       [2026-02-11 00:53:08,501612][I][ezpz/pbs:273:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
       [2026-02-11 00:53:08,502360][I][ezpz/launch:391:build_executable] Building command to execute by piecing together:
       [2026-02-11 00:53:08,502775][I][ezpz/launch:392:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12461964.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
       [2026-02-11 00:53:08,503436][I][ezpz/launch:393:build_executable] (2.) cmd_to_launch: python3 -m torchtitan.experiments.ezpz.train --model.name ezpz.agpt --model.flavor 2b --training.seq_len 8192 --training.local_batch_size=1 --model.hf_assets_path assets/hf/gemma-7b --job.print_config --compile.enable --training.global_batch_size=48 --metrics.log_freq=1 --metrics.enable_wandb
         
       # ...clipped...

       wandb: Tracking run with wandb version 0.24.2
       wandb: Run data is saved locally in outputs/tb/20260211-0053/wandb/run-20260211_005323-pq1k68c0
       wandb: Run `wandb offline` to turn off syncing.
       wandb: Syncing run ancient-aardvark-62
       wandb:  View project at https://wandb.ai/aurora_gpt/torchtitan
       wandb:  View run at https://wandb.ai/aurora_gpt/torchtitan/runs/pq1k68c0
       [titan] 2026-02-11 00:53:31,804 - root - INFO - WandB logging enabled
       [titan] 2026-02-11 00:53:31,810 - root - INFO - XPU capacity: Intel(R) Data Center GPU Max 1550 with 63.98GiB memory
       [titan] 2026-02-11 00:53:31,856 - __main__ - INFO - Model ezpz.agpt 2b size: 1,703,462,912 total parameters
       [titan] 2026-02-11 00:53:31,857 - root - INFO - Compiling the loss function with torch.compile
       [titan] 2026-02-11 00:53:31,867 - root - INFO - Applied selective activation checkpointing to the model
       [titan] 2026-02-11 00:53:31,871 - root - INFO - Compiling each TransformerBlock with torch.compile
       [titan] 2026-02-11 00:53:31,900 - root - INFO - Applied FSDP to the model
       [titan] 2026-02-11 00:53:32,274 - __main__ - INFO - Peak FLOPS used for computing MFU: 2.982e+14
       [titan] 2026-02-11 00:53:32,275 - __main__ - INFO - XPU memory usage for model: 0.30GiB(0.48%)

       # ...clipped...
         
       [titan] 2026-02-11 00:54:17,768 - root - INFO - step:  2  loss: 12.85991  grad_norm:  2.1345  memory: 19.45GiB(30.39%)  tps: 5,941  tflops: 56.37  mfu: 18.91%
       [titan] 2026-02-11 00:54:17,777 - root - WARNING - Dataset c4_test is being re-looped
       [titan] 2026-02-11 00:54:20,545 - root - INFO - step:  3  loss: 12.66468  grad_norm:  2.1721  memory: 19.45GiB(30.39%)  tps: 5,902  tflops: 56.01  mfu: 18.78%
       [titan] 2026-02-11 00:54:23,317 - root - INFO - step:  4  loss: 12.37955  grad_norm:  2.1435  memory: 19.45GiB(30.39%)  tps: 5,914  tflops: 56.12  mfu: 18.82%
       [titan] 2026-02-11 00:54:23,332 - root - WARNING - Dataset c4_test is being re-looped
       [titan] 2026-02-11 00:54:26,101 - root - INFO - step:  5  loss: 12.10499  grad_norm:  2.9399  memory: 19.45GiB(30.39%)  tps: 5,888  tflops: 55.88  mfu: 18.74%
       [titan] 2026-02-11 00:54:28,875 - root - INFO - step:  6  loss: 11.76838  grad_norm:  3.3020  memory: 19.45GiB(30.39%)  tps: 5,909  tflops: 56.07  mfu: 18.80%
       [titan] 2026-02-11 00:54:28,896 - root - WARNING - Dataset c4_test is being re-looped
       [titan] 2026-02-11 00:54:31,642 - root - INFO - step:  7  loss: 11.37482  grad_norm:  4.0258  memory: 19.45GiB(30.39%)  tps: 5,924  tflops: 56.21  mfu: 18.85%
       [titan] 2026-02-11 00:54:34,410 - root - INFO - step:  8  loss: 10.82563  grad_norm:  4.0790  memory: 19.45GiB(30.39%)  tps: 5,920  tflops: 56.18  mfu: 18.84%
       [titan] 2026-02-11 00:54:37,180 - root - INFO - step:  9  loss: 10.44119  grad_norm:  6.9668  memory: 19.45GiB(30.39%)  tps: 5,919  tflops: 56.17  mfu: 18.84%
       [titan] 2026-02-11 00:54:37,185 - root - WARNING - Dataset c4_test is being re-looped
       [titan] 2026-02-11 00:54:39,941 - root - INFO - step: 10  loss: 10.37038  grad_norm: 13.6683  memory: 19.45GiB(30.39%)  tps: 5,935  tflops: 56.32  mfu: 18.89%
       [titan] 2026-02-11 00:54:42,702 - root - INFO - step: 11  loss: 10.03176  grad_norm:  6.8594  memory: 19.45GiB(30.39%)  tps: 5,937  tflops: 56.34  mfu: 18.89%
       [titan] 2026-02-11 00:54:42,716 - root - WARNING - Dataset c4_test is being re-looped
       [titan] 2026-02-11 00:54:45,467 - root - INFO - step: 12  loss:  9.81982  grad_norm:  7.2047  memory: 19.45GiB(30.39%)  tps: 5,928  tflops: 56.26  mfu: 18.87%
       [titan] 2026-02-11 00:54:48,223 - root - INFO - step: 13  loss:  9.69024  grad_norm:  5.0144  memory: 19.45GiB(30.39%)  tps: 5,947  tflops: 56.43  mfu: 18.92%
       [titan] 2026-02-11 00:54:48,238 - root - WARNING - Dataset c4_test is being re-looped
       [titan] 2026-02-11 00:54:50,979 - root - INFO - step: 14  loss:  9.55574  grad_norm:  3.9267  memory: 19.45GiB(30.39%)  tps: 5,947  tflops: 56.44  mfu: 18.93%
       [titan] 2026-02-11 00:54:53,734 - root - INFO - step: 15  loss:  9.34104  grad_norm:  3.2386  memory: 19.45GiB(30.39%)  tps: 5,949  tflops: 56.45  mfu: 18.93%
       [titan] 2026-02-11 00:54:56,490 - root - INFO - step: 16  loss:  9.14187  grad_norm:  2.7580  memory: 19.45GiB(30.39%)  tps: 5,948  tflops: 56.44  mfu: 18.93%
       [titan] 2026-02-11 00:54:56,495 - root - WARNING - Dataset c4_test is being re-looped
       [titan] 2026-02-11 00:54:59,248 - root - INFO - step: 17  loss:  9.03939  grad_norm:  3.9063  memory: 19.45GiB(30.39%)  tps: 5,944  tflops: 56.40  mfu: 18.91%
       [titan] 2026-02-11 00:55:02,006 - root - INFO - step: 18  loss:  8.82981  grad_norm:  3.2502  memory: 19.45GiB(30.39%)  tps: 5,942  tflops: 56.39  mfu: 18.91%
       [titan] 2026-02-11 00:55:02,019 - root - WARNING - Dataset c4_test is being re-looped
       [titan] 2026-02-11 00:55:04,767 - root - INFO - step: 19  loss:  8.66831  grad_norm:  2.4284  memory: 19.45GiB(30.39%)  tps: 5,936  tflops: 56.33  mfu: 18.89%
       ```

     </details>


[^rc4]:
    This was run while testing the final release candidate for the new
    `frameworks` module which will soon\* be deployed on Aurora at ALCF:

    ```bash
    module add hdf5
    module add pti-gpu

    export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"

    source /opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/miniforge3-25.11.0-1-uydwzvt/bin/activate
    conda activate /lus/tegu/projects/datasets/software/26.26.0/wheelforge/envs/frameworks_install/frameworks_2025.3.1-RC4 ; export ZE_FLAT_DEVICE_HIERARCHY=FLAT ; export CCL_PROCESS_LAUNCHER=pmix ; export CCL_OP_SYNC=1
    ```

    \* soon as of 02/11/2026
     

## References


- 🍋 `ezpz`:
  - Documentation: [ezpz.cool](https://ezpz.cool)
  - GitHub: [saforem2/ezpz](https://github.com/saforem2/ezpz)







