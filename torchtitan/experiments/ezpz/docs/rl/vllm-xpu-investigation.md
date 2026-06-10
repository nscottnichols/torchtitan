# vLLM-XPU on torch 2.13 — investigation findings

**Date:** 2026-06-10
**Goal:** Determine whether we can replace the current all-ranks-generate
HF `.generate()` path in `ezpz/rl` with a vLLM-backed
generator/trainer split, **without** breaking our torch 2.13 XPU
build in `.venv/`.

## TL;DR

**Blocked on a libtorch C++ ABI mismatch.** vLLM 0.22.1 imports
cleanly on our torch 2.13.0.dev+xpu build (sibling venv, all deps
installed `--no-deps`), but the `vllm_xpu_kernels` companion package
that vLLM requires for the XPU platform path is pre-built against
torch 2.12 and references C++ symbols (e.g.
`c10::impl::cow::materialize_cow_storage`) that don't exist in
torch 2.13. Building `vllm-xpu-kernels` from source against our
torch would unblock this — estimated 1–2 days of work given the
oneAPI/SYCL build chain.

Until then, the all-ranks-`.generate()` design in `ezpz/rl` stays.
It's correct, just slow.

## What I tried

### 1. Install vLLM into a sibling venv (`venvs/vllm-test/`)

Used `uv venv` + `uv pip install vllm --no-deps` so vLLM couldn't
pull a different (CUDA-only) torch over our `.venv/`'s torch 2.13.

Then walked the missing-pure-python-dep tree via an auto-loop until
`from vllm import LLM, SamplingParams` succeeded:

```
distro, cloudpickle, msgspec, pyzmq, fastapi, uvicorn, openai,
pydantic, prometheus-client, tiktoken, sentencepiece, protobuf,
psutil, py-cpuinfo, cbor2, gguf, einops, blake3, partial-json-parser,
outlines, outlines-core, lm-format-enforcer, diskcache,
mistral-common, ray, starlette, anyio, sniffio, h11, httptools,
websockets, watchfiles, python-multipart, openai-harmony,
pybase64, cachetools, uvloop
```

Total ~36 pure-Python pip packages, ~30s install in aggregate. None
of them pull torch as a hard dep (we kept `--no-deps` throughout).

**Result:** vLLM imports work end-to-end on our torch 2.13 from a
sibling venv with `PYTHONPATH=.venv/lib/python3.14/site-packages`.

### 2. Platform detection on a compute node

ssh'd into one of my eval-allocation nodes (with XPU devices
visible) and ran vLLM's platform-detection probe:

```python
from vllm.platforms import current_platform
print(type(current_platform).__name__)   # → UnspecifiedPlatform
print(current_platform.is_xpu())          # → False
```

`torch.xpu.is_available()` returns `True` and `torch.xpu.device_count()`
returns 1, but vLLM still falls through to `UnspecifiedPlatform`.

With `VLLM_LOGGING_LEVEL=DEBUG`:

```
Checking if XPU platform is available.
XPU platform is not available because: No module named 'vllm_xpu_kernels'
```

### 3. Install `vllm-xpu-kernels`

Active upstream project at
[vllm-project/vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)
(last update **today, 2026-06-10**). Releases publish a single
`vllm_xpu_kernels-<ver>-cp38-abi3-manylinux_2_28_x86_64.whl`
(stable Python ABI, ~350 MB).

Downloaded `v0.1.9.1` (2026-06-01), installed with `--no-deps`. The
`cp38-abi3` Python ABI is broad (works on cpython 3.8 through 3.14)
— but Python ABI compatibility does **not** imply libtorch ABI
compatibility.

### 4. Retry platform detection — symbol-level failure

```
XPU platform is not available because:
.../vllm_xpu_kernels/_C.abi3.so:
undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
```

Demangled: `c10::impl::cow::materialize_cow_storage(c10::StorageImpl&)`.

Verified the symbol does not exist in our
`.venv/lib/python3.14/site-packages/torch/lib/libtorch_cpu.so`:

```bash
$ nm -D .venv/.../libtorch_cpu.so | grep materialize_cow_storage
(no output)
```

The pre-built wheel was linked against a torch version that exposes
this symbol — most likely the upstream-pinned `torch==2.12.0+xpu`
that `vllm-xpu-kernels`'s README requires:

> **PyTorch**: 2.12.0+xpu

`cp38-abi3` is the **Python** stable ABI tag. It's about the
Python/C API surface, not about libtorch (`libtorch_cpu.so`,
`libc10.so`). The kernels wheel links against libtorch at build
time, so it's effectively pinned to whatever torch version the
maintainers used for the build.

## Why this matters

Without the kernels, vLLM falls back to a CPU-only path on this
host. Useful for sanity checks but not for the 5–10× generator
speedup we want vs the current HF `.generate()` design.

## Paths forward

**Option A — Build `vllm-xpu-kernels` from source against our torch 2.13.**
Requires the oneAPI 2025.3 + CMake ≥ 3.26 + Ninja toolchain
(already on the system) plus SYCL/DPC++ knowledge. The repo's
build instructions are minimal but appear to be standard
PyTorch-style C++ extension builds. Estimated 1–2 days of trial,
mostly debugging template instantiations against newer torch
headers.

**Option B — Pin a sibling venv to vLLM's required `torch==2.12.0+xpu`.**
Single-line `uv pip install torch==2.12.0+xpu --extra-index-url
https://download.pytorch.org/whl/xpu` into `venvs/vllm-test/`.
Doesn't touch `.venv/`. The generator process would run on torch
2.12 while the trainer runs on torch 2.13. Acceptable because the
generator only does forward passes / weight loads — no autograd
state shared across processes — but does mean two different torch
builds on the same machine, with the version skew risks that
implies.

**Option C — Defer until vLLM ships a torch 2.13 wheel.**
`vllm-xpu-kernels` is actively maintained (13 releases since
2026-01-29, latest 2026-06-01). They'll bump to torch 2.13 once
that's stable upstream. Could be weeks, could be months.

**Recommendation: Option B.** Lowest engineering cost (~10 min),
clean separation between trainer and generator processes, no
need to wait for upstream. The cost is maintaining two torch
versions in two venvs — annoying but tractable. Worth the trade
if we want a usable vLLM generator before the upstream catches up.

## Sibling venv inventory (created during this investigation)

| Path | Contents | Status |
|---|---|---|
| `venvs/vllm-test/` | vLLM 0.22.1 (no-deps) + 37 pure-py runtime deps + vllm-xpu-kernels 0.1.9.1 | broken — kernels wheel needs torch 2.12, ours is 2.13 |
| `/tmp/vllm-xpu-wheel/vllm_xpu_kernels-0.1.9.1-cp38-abi3-manylinux_2_28_x86_64.whl` | cached 350 MB wheel from upstream release | reusable for option A debugging |

## In the meantime

The all-ranks-generate design in `ezpz/rl` stays — see
[`docs/rl/README.md`](README.md) "Limitations" section. It's
provably correct (same reward, same gradient as a split design),
just leaves a 5–10× generation throughput improvement on the
floor. For first-cut production GRPO runs (e.g. 1000 steps on
arithmetic at 8N), this is fine; we burn extra compute but the
experiments still finish in walltime. Revisit when GRPO becomes a
larger fraction of total compute than SFT or pretraining.
