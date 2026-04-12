# 80B Benchmark

## Aurora Results


| Field        | Value                                                  |
| ------------ | ------------------------------------------------------ |
| Date         | 2026-03-30T06:54:54-05:00                              |
| Commit       | c6ff706                                                |
| Machine      | x1921c1s2b0n0                                          |
| Job ID       | 12463902.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov |
| Nodes        | 2                                                      |
| Devices      | 24                                                     |
| Devices/Node | 12                                                     |
| Steps        | 5                                                      |

### Aurora Runs


| Model        | NGPU | TP | PP | DP | GBS | Memory           | TPS | TFLOPS | MFU   | Wall (s) | W&B                                                                     | Status |
|--------------|------|----|----|----|-----|------------------|-----|--------|-------|----------|:-----------------------------------------------------------------------:|:------:|
| 80B          | 24   | 2  | 1  | 12 | 12  | 59.82GiB(93.49%) | 85  | 46.31  | 15.53 | 331      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/qks5kewd) | OK     |
| 80B_alt      | 24   | 2  | 1  | 12 | 12  | 59.82GiB(93.49%) | 81  | 44.34  | 14.87 | 326      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/t60yimw5) | OK     |
| 80B_alt      | 24   | 3  | 1  | 8  | 8   | 51.61GiB(80.66%) | 23  | 12.40  | 4.16  | 694      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/mna9aesf) | OK     |
| 80B_alt      | 24   | 6  | 1  | 4  | 4   | 39.54GiB(61.79%) | 43  | 23.30  | 7.82  | 233      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/3j7ibvae) | OK     |
| 80B_alt      | 24   | 12 | 1  | 2  | 2   | N/A              | N/A | N/A    | N/A   | 23       |                                                                         | CRASH  |
| 80B_wide     | 24   | 2  | 1  | 12 | 12  | 60.75GiB(94.94%) | 43  | 22.23  | 7.45  | 1002     | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/i0fbuhab) | OOM    |
| 80B_wide     | 24   | 3  | 1  | 8  | 8   | 52.75GiB(82.44%) | 29  | 15.16  | 5.08  | 559      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/6vzikdog) | OK     |
| 80B_wide     | 24   | 6  | 1  | 4  | 4   | 40.47GiB(63.26%) | 51  | 26.35  | 8.84  | 216      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ltaakcpx) | OK     |
| 80B_wide     | 24   | 12 | 1  | 2  | 2   | N/A              | N/A | N/A    | N/A   | 22       |                                                                         | CRASH  |
| 80B_deep     | 24   | 2  | 1  | 12 | 12  | 58.84GiB(91.95%) | 83  | 45.58  | 15.29 | 324      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/er9y8co7) | OK     |
| 80B_deep_alt | 24   | 2  | 1  | 12 | 12  | 58.95GiB(92.14%) | 77  | 41.87  | 14.04 | 343      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/924exg6x) | OK     |
| 80B_deep_alt | 24   | 3  | 1  | 8  | 8   | 47.43GiB(74.13%) | 23  | 12.56  | 4.21  | 690      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/j7xpn4cr) | OK     |
| 80B_deep_alt | 24   | 6  | 1  | 4  | 4   | 41.63GiB(65.06%) | 41  | 22.43  | 7.52  | 245      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/okgm1auw) | OK     |
| 80B_deep_alt | 24   | 12 | 1  | 2  | 2   | 35.48GiB(55.45%) | 28  | 15.40  | 5.16  | 210      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/bea2k4u0) | OK     |

Logs:
`outputs/benchmarks/80b_20260330_065453/`

## Sunspot Results

| Field        | Value                                                  |
|--------------|--------------------------------------------------------|
| Date         | 2026-03-30T06:54:54-05:00                              |
| Commit       | c6ff706                                                |
| Machine      | x1921c1s2b0n0                                          |
| Job ID       | 12463902.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov |
| Nodes        | 2                                                      |
| Devices      | 24                                                     |
| Devices/Node | 12                                                     |
| Steps        | 5                                                      |

### Sunspot Runs


| Model        | TP | PP | DP | Memory           | TPS | TFLOPS | MFU   | Wall (s) | W&B                                                                     | Status |
|--------------|----|----|----|------------------|-----|--------|-------|----------|:-----------------------------------------------------------------------:|:------:|
| 80B          | 2  | 1  | 12 | 59.82GiB(93.49%) | 85  | 46.31  | 15.53 | 331      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/qks5kewd) | OK     |
| 80B_alt      | 2  | 1  | 12 | 59.82GiB(93.49%) | 81  | 44.34  | 14.87 | 326      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/t60yimw5) | OK     |
| 80B_alt      | 3  | 1  | 8  | 51.61GiB(80.66%) | 23  | 12.40  | 4.16  | 694      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/mna9aesf) | OK     |
| 80B_alt      | 6  | 1  | 4  | 39.54GiB(61.79%) | 43  | 23.30  | 7.82  | 233      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/3j7ibvae) | OK     |
| 80B_alt      | 12 | 1  | 2  | XXX              | XXX | XXX    | XXX   | 23       | NONE                                                                    | CRASH  |
| 80B_wide     | 2  | 1  | 12 | 60.75GiB(94.94%) | 43  | 22.23  | 7.45  | 1002     | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/i0fbuhab) | OOM    |
| 80B_wide     | 3  | 1  | 8  | 52.75GiB(82.44%) | 29  | 15.16  | 5.08  | 559      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/6vzikdog) | OK     |
| 80B_wide     | 6  | 1  | 4  | 40.47GiB(63.26%) | 51  | 26.35  | 8.84  | 216      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ltaakcpx) | OK     |
| 80B_wide     | 12 | 1  | 2  | XXX              | XXX | XXX    | XXX   | 22       | NONE                                                                    | CRASH  |
| 80B_deep     | 2  | 1  | 12 | 58.84GiB(91.95%) | 83  | 45.58  | 15.29 | 324      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/er9y8co7) | OK     |
| 80B_deep_alt | 2  | 1  | 12 | 58.95GiB(92.14%) | 77  | 41.87  | 14.04 | 343      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/924exg6x) | OK     |
| 80B_deep_alt | 3  | 1  | 8  | 47.43GiB(74.13%) | 23  | 12.56  | 4.21  | 690      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/j7xpn4cr) | OK     |
| 80B_deep_alt | 6  | 1  | 4  | 41.63GiB(65.06%) | 41  | 22.43  | 7.52  | 245      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/okgm1auw) | OK     |
| 80B_deep_alt | 12 | 1  | 2  | 35.48GiB(55.45%) | 28  | 15.40  | 5.16  | 210      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/bea2k4u0) | OK     |


Logs:
`outputs/benchmarks/80b_20260330_065453/`
