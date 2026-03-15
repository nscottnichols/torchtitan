# MoE Training Configs

## Available Models

| Config           | Model                                    | Description                   |
| ---------------- | ---------------------------------------- | ----------------------------- |
| `moe_debugmodel` | debugmodel (256d, 6L, 8 experts)         | Tiny model for fast iteration |
| `moe_small`      | small (2048d, 24L, 64 experts)           | Small model                   |
| `moe_10b_2b`     | 10B/2B (2048d, 27L, 36 experts, top_k=3) | 10B total / 2B active         |
| `moe_16b`        | 16B (2048d, 27L, 64 experts)             | Full 16B                      |
| `moe_671b`       | 671B (7168d, 61L, 256 experts)           | Full 671B                     |

Each has a `_from_json` variant (e.g. `moe_10b_2b_from_json`) that applies
overrides from the `TT_CONFIG_JSON` environment variable

---

## Using JSON Override Configs

Set `TT_CONFIG_JSON` to a JSON file, then use a `_from_json` config:

```bash
TT_CONFIG_JSON=<path/to/config.json> \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json
```

### Aurora (2 nodes, 24 XPUs, EP=12)

Pre-configured JSON overrides: `moe_runs/`

```bash
# 1. Smoke test (40 steps, seq_len=1024)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_2nodes_smoke.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 2. Baseline (1000 steps, seq_len=4096)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_2nodes.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 3. Throughput test (seq_len=4096, gc_freq=200)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_2nodes_4096_perf.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 4. Prod sim — no AC (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 5. Prod sim + selective AC op-level (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim_ac.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 6. Selective AC op-level, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim_ac_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 7. Full AC, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim_ac_full_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 8. Layer-1 only AC, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim_ac_layer1_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 9. Selective AC + compile FFN+loss (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim_ac_lb2_compile_ffn.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 10. 128-node scale run (1536 XPUs)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_128nodes_4096_prod_sim_ac_lb2_compile_ffn.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable
```

### Polaris (2 nodes, 8 GPUs)

Pre-configured JSON overrides: `moe_runs/polaris/`

```bash
# 1. Smoke test (40 steps, seq_len=1024)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris/deepseek_v3_10b2b_polaris_2nodes_smoke.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 2. Baseline (1000 steps, seq_len=4096)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris/deepseek_v3_10b2b_polaris_2nodes.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 3. Throughput test (seq_len=4096, gc_freq=200)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris/deepseek_v3_10b2b_polaris_2nodes_4096_perf.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 4. Prod sim — no AC (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 5. Prod sim + selective AC op-level (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim_ac.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 6. Selective AC op-level, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim_ac_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 7. Full AC, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim_ac_full_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 8. Layer-1 only AC, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim_ac_layer1_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 9. Selective AC + compile FFN+loss (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim_ac_lb2_compile_ffn.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable
```

### Writing Custom JSON Overrides

Any trainer config field can be overridden. Nest by config section:

```json
{
  "training": {
    "local_batch_size": 4,
    "seq_len": 2048,
    "steps": 500
  },
  "parallelism": {
    "data_parallel_replicate_degree": 4,
    "data_parallel_shard_degree": -1
  },
  "activation_checkpoint": {
    "mode": "selective",
    "selective_ac_option": "op"
  }
}
```

Unknown fields raise `KeyError`, type mismatches raise `TypeError`.

---

## Using CLI Flags Directly (No JSON)

Use `--config moe_10b_2b` (without `_from_json`) and pass all overrides as CLI flags.
The `$(ezpz_get_machine_name)` shell expansion auto-selects `aurora` or `polaris`.

For Aurora EP=12 topology, add:
`--parallelism.expert_parallel_degree 12 --parallelism.data_parallel_shard_degree 12`

```bash
# 1. Smoke test (40 steps, seq_len=1024)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 1 --training.seq_len 1024 --training.steps 40 \
  --metrics.log_freq 1 \
  --dataloader.dataset blendcorpus \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 2. Baseline (1000 steps, seq_len=4096)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 1 --training.seq_len 4096 --training.steps 1000 \
  --metrics.log_freq 1 \
  --dataloader.dataset blendcorpus \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 3. Throughput test (seq_len=4096, gc_freq=200)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 1 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 200 --metrics.log_freq 10 \
  --dataloader.dataset blendcorpus \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 4. Prod sim — no AC (local_batch=2)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 2 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode none \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 5. Prod sim + selective AC op-level (local_batch=2)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 2 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option op \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 6. Selective AC op-level, higher batch (local_batch=3)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 3 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option op \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 7. Full AC, higher batch (local_batch=3)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 3 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode full \
  --activation_checkpoint.no_preserve_rng_state \
  --activation_checkpoint.determinism_check none \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 8. Layer-1 only AC, higher batch (local_batch=3)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 3 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option 1 \
  --activation_checkpoint.no_preserve_rng_state \
  --activation_checkpoint.determinism_check none \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 9. Selective AC + compile FFN+loss (local_batch=2)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 2 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option op \
  --compile.enable --compile.components feed_forward loss --compile.backend inductor \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt
```
