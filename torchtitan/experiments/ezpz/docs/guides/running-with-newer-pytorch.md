# Running with Newer PyTorch (>= 2.10)

> [!NOTE]
> We will use the alias `uvi`:
>
> ```bash
> alias uvi='uv pip install --no-cache --link-mode=copy
> ```

1. Clone torchtitan:

   ```bash
   gh repo clone saforem2/torchtitan -- --branch ezpz
   cd torchtitan
   ```

2. Load modules and export environment variables:

   ```bash
   module load oneapi/release/2025.3.1 hdf5 pti-gpu
   export ZE_FLAT_DEVICE_HIERARCHY=FLAT
   export CCL_PROCESS_LAUNCHER=pmix
   export CCL_OP_SYNC=1
   export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
   export TORCH_CPP_LOG_LEVEL=ERROR
   ```

3. Create venv:

   ```bash
   uv venv --python=3.12
   source .venv/bin/activate
   ```

4. Install PyTorch:

   ```bash
   uvi torch torchvision torchaudio torchdata \
       --pre \
       --index-url https://download.pytorch.org/whl/nightly/xpu \
       --upgrade
   ```

5. Install dependencies:

   ```bash
   uvi torchcomms tyro tensorboard deepspeed mpi4py
   uvi "git+https://github.com/zhenghh04/blendcorpus"
   uvi "git+https://github.com/saforem2/ezpz@fix/mfu-tracking"
   ```

6. Remove Intel's MPI runtime (`impi-rt`):

   ```bash
   uv pip uninstall impi-rt
   ```

7. Download tokenizers:

   ```bash
   python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
   ```

8. (Optional) Copy `.venv` to `/tmp/.venv` on all nodes
   (significantly faster startup, better performance):

   ```bash
   ezpz yeet-env
   ```

   1. Deactivate current venv:

      ```bash
      deactivate
      ```

   2. Activate from `/tmp/.venv/`:

      ```bash
      source /tmp/.venv/bin/activate
      ```

9. Run training:

   ```bash
   FLAVOR=agpt
   CONFIG=debugmodel
   ezpz launch python3 -m torchtitan.experiments.ezpz.train \
       --module="ezpz.${FLAVOR}" \
       --config="${FLAVOR}_${CONFIG}" \
       --checkpoint.no-enable \
       --training.steps=10
   ```
