# Running with Newer PyTorch (>= 2.10)

> [!NOTE]
> We will use the alias `uvi`:
>
> ```bash
> alias uvi='uv pip install --no-cache --link-mode=copy'
> ```

1. Clone torchtitan:

   ```bash
   gh repo clone saforem2/torchtitan -- --branch ezpz
   cd torchtitan
   ```

1. Load modules and export environment variables:

   ```bash
   module load oneapi/release/2025.3.1 hdf5 pti-gpu
   export ZE_FLAT_DEVICE_HIERARCHY=FLAT
   export CCL_PROCESS_LAUNCHER=pmix
   export CCL_OP_SYNC=1
   export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
   export TORCH_CPP_LOG_LEVEL=ERROR
   ```

1. Create venv:

   ```bash
   uv venv --python=3.14
   source .venv/bin/activate
   ```

1. Install PyTorch:

   ```bash
   uvi torch torchvision torchaudio torchdata \
       --pre \
       --index-url https://download.pytorch.org/whl/nightly/xpu \
       --upgrade
   ```

1. Install dependencies:

   ```bash
   uvi torchcomms tyro tensorboard deepspeed mpi4py
   uvi "git+https://github.com/zhenghh04/blendcorpus"
   uvi "git+https://github.com/saforem2/ezpz"
   ```

1. Remove Intel's MPI runtime (`impi-rt`):

   ```bash
   uv pip uninstall impi-rt
   ```

1. Download tokenizers:

   ```bash
   python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
   ```

1. Run training:

   ```bash
   FLAVOR=agpt
   CONFIG=debugmodel
   ezpz launch python3 -m torchtitan.experiments.ezpz.train \
       --module="ezpz.${FLAVOR}" \
       --config="${FLAVOR}_${CONFIG}" \
       --checkpoint.no-enable \
       --training.steps=10
   ```

## Running at Large Scale (> 512 nodes)

At a few nodes, importing Python from a shared filesystem is a small tax —
~milliseconds per import, paid once.

At 6k workers all hammering Lustre with the same import waterfall, that small
tax becomes minutes of dead time before the first training step lands, plus
tail-latency stragglers that dominate collective wait time for the rest of the
run.

[`ezpz yeet`](https://ezpz.cool/cli/yeet/) sidesteps this by copying the active
venv (or a pre-built `.venv.tar.gz`) to node-local `/tmp/` storage on every
worker, so subsequent imports, checkpoint loads, and config reads hit local SSD
instead of Lustre.

The local copy is patched once on the source node (activate scripts, shebangs,
symlinks) and then distributed via a greedy `rsync` fan-out — each completed
node immediately becomes a source for others, so the broadcast tree grows in
roughly `O(log N)` time instead of saturating one NIC.

### Measured scaling on Aurora (8 → 4096 nodes)


| Nodes | yeet (s) | First-step (s) | Per-node (ms) |
| ----: | -------: | -------------: | ------------: |
|     8 |     69.7 |           29.3 |         8,712 |
|    64 |     91.2 |           34.6 |         1,425 |
|   512 |    174.5 |           44.5 |           341 |
|  1024 |    255.4 |           60.8 |           249 |
|  2048 |    421.4 |           94.8 |           206 |
|  4096 |    750.6 |          194.0 |           183 |


Two regimes:

- < 128 nodes the cost is dominated by the one-time local extract (~70-91 s
  flat)
- ≥ 128 nodes the broadcast tree depth and per-leaf contention dominate, with
  each 2× in nodes adding ~1.5-1.8× wall-clock.

Even at full-Aurora 4096-node scale the pre-launch overhead is under 13 minutes,
versus the 1-2 hours the per-file rsync mode was projected to take.

For more detail (full sweep, plots, methodology) see the
[yeet CLI docs](https://ezpz.cool/cli/yeet/) and the
[benchmark harness](https://github.com/saforem2/torchtitan/tree/ezpz/torchtitan/experiments/ezpz/docs/scaling/yeet_env).

### Workflow

> [!TIP]
> If you already built a tarball with `ezpz tar-env`, pass it explicitly —
> tarball broadcast is ~10× faster than per-file `rsync` at scale because the
> Lustre side becomes one sequential read instead of millions of `stat()`s.
> Plain `ezpz yeet` will print a hint when it sees a same-named `.tar.gz`
> sitting nearby.

1. Be sure to load the appropriate modules and activate the `.venv` we just
   created:

   ```bash
   source <(curl -fsSL https://bit.ly/ezpz-utils)
   ezpz_setup_job
   ezpz_setup_xpu
   source .venv/bin/activate
   ```

1. Create a compressed tarball (`.tar.gz`) from the `.venv`:

   ```bash
   ezpz tar-env
   # or, alternatively:
   # tar czvf .venv.tar.gz --directory .venv .
   ```

   Note:
   This will take a few minutes but only needs to be done _once_; after that, we
   can simply reuse the `.venv.tar.gz` for subsequent runs[^tarball].

1. Distribute the tarball to every node's `/tmp/`:

   ```bash
   ezpz yeet .venv.tar.gz   # faster at scale

   # alternatively, yeet the uncompressed `.venv/`:
   # ezpz yeet              # slower at scale
   ```

1. Switch to the local copy:

   ```bash
   deactivate
   source /tmp/.venv/bin/activate
   ```

1. Launch training as usual; `ezpz launch` respects `$VIRTUAL_ENV`, so it picks
   up the `/tmp/` venv automatically:

   ```bash
   ezpz launch python3 -m torchtitan.experiments.ezpz.train \
       --module="ezpz.${FLAVOR}" \
       --config="${FLAVOR}_${CONFIG}" \
       --checkpoint.no-enable \
       --training.steps=10
   ```

> [!IMPORTANT]
> `/tmp/` is node-local.
> Keep your project directory (data, checkpoints) on a shared filesystem so
> all ranks can read inputs and write outputs.

[^tarball]:
If we install additional packages or make changes to the `.venv/`, we will need
to repeat this step and create a new `.venv.tar.gz` to capture these.
