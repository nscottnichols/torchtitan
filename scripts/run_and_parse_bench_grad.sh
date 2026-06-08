#!/usr/bin/env bash
set -euo pipefail

LOG="${1:-/tmp/tt_moe_bench_grad_$(date +%Y%m%d_%H%M%S).log}"

if [[ ! -x ./scripts/run_bench_10b_2b_sdpa_xpu.sh ]]; then
  echo "ERROR: run this from the torchtitan repo root with scripts/run_bench_10b_2b_sdpa_xpu.sh present." >&2
  exit 1
fi

# Grad-enabled MoE microbenchmark for training-relevant dispatcher/combine paths.
# Defaults are lighter than the forward-only benchmark because this includes
# backward and stores activations. Override WARMUP/ITERS/BATCH_SIZE/SEQ_LEN as
# needed when comparing variants.
export WARMUP="${WARMUP:-5}"
export ITERS="${ITERS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export SEQ_LEN="${SEQ_LEN:-2048}"

time ./scripts/run_bench_10b_2b_sdpa_xpu.sh --backward 2>&1 | tee "${LOG}"

echo
echo "== parsed result =="
grep -E \
  'benchmark=|model_flavor=|mode=|device=|backend=|world_size=|dp_degree=|ep_degree=|batch_size_per_rank=|seq_len=|dim=|hidden_dim=|num_experts_global=|num_shared_experts=|top_k=|dtype=|score_func=|score_before_experts=|force_load_balance=|use_grouped_mm=|include_ffn_norm=|compile=|forward_context=|loss=|optimizer_step=|patch_xpu_a2a=|slowest_rank_mean_ms=|global_tokens_per_sec=|p50=|p90=|p99=|TT_MOE_FASTPATH_COUNTERS' \
  "${LOG}" || true

echo "log=${LOG}"
