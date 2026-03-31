# AuroraGPT-80B Benchmark

> Sam Foreman  
> 2026-03-31

## AuroraGPT Dense Model Configs

| Variant      | HIDDEN | NLAYERS | HEADS | KV_HEADS | FFN_HIDDEN |
|--------------|--------|---------|-------|----------|------------|
| 2B           | 2048   | 12      | 16    | 4        | 11008      |
| 20B          | 5120   | 64      | 40    | 8        | 14336      |
| 80B          | 9216   | 84      | 72    | 12       | 25600      |
| 80B_alt      | 9216   | 84      | 72    | 12       | 25596      |
| 80B_wide     | 10752  | 48      | 84    | 12       | 39936      |
| 80B_deep     | 7680   | 96      | 60    | 12       | 28672      |
| 80B_deep_alt | 7680   | 96      | 60    | 12       | 28668      |


## Aurora Environment


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


| Status | Model        | NGPU | TP | PP | DP | GBS | Memory           | TPS | TFLOPS | MFU   | Wall (s) | W&B                  |
|:------:|--------------|------|----|----|----|-----|------------------|-----|--------|-------|----------|:--------------------:|
| ✅     | 80B          | 24   | 2  | 1  | 12 | 12  | 59.82GiB(93.49%) | 85  | 46.31  | 15.53 | 331      | [qks5kewd][qks5kewd] |
| ✅     | 80B_alt      | 24   | 2  | 1  | 12 | 12  | 59.82GiB(93.49%) | 81  | 44.34  | 14.87 | 326      | [t60yimw5][t60yimw5] |
| ✅     | 80B_alt      | 24   | 3  | 1  | 8  | 8   | 51.61GiB(80.66%) | 23  | 12.40  | 4.16  | 694      | [mna9aesf][mna9aesf] |
| ✅     | 80B_alt      | 24   | 6  | 1  | 4  | 4   | 39.54GiB(61.79%) | 43  | 23.30  | 7.82  | 233      | [3j7ibvae][3j7ibvae] |
| ❌     | 80B_alt      | 24   | 12 | 1  | 2  | 2   | N/A              | N/A | N/A    | N/A   | 23       |                      |
| ☠️     | 80B_wide     | 24   | 2  | 1  | 12 | 12  | 60.75GiB(94.94%) | 43  | 22.23  | 7.45  | 1002     | [i0fbuhab][i0fbuhab] |
| ✅     | 80B_wide     | 24   | 3  | 1  | 8  | 8   | 52.75GiB(82.44%) | 29  | 15.16  | 5.08  | 559      | [6vzikdog][6vzikdog] |
| ✅     | 80B_wide     | 24   | 6  | 1  | 4  | 4   | 40.47GiB(63.26%) | 51  | 26.35  | 8.84  | 216      | [ltaakcpx][ltaakcpx] |
| ❌     | 80B_wide     | 24   | 12 | 1  | 2  | 2   | N/A              | N/A | N/A    | N/A   | 22       |                      |
| ✅     | 80B_deep     | 24   | 2  | 1  | 12 | 12  | 58.84GiB(91.95%) | 83  | 45.58  | 15.29 | 324      | [er9y8co7][er9y8co7] |
| ✅     | 80B_deep_alt | 24   | 2  | 1  | 12 | 12  | 58.95GiB(92.14%) | 77  | 41.87  | 14.04 | 343      | [924exg6x][924exg6x] |
| ✅     | 80B_deep_alt | 24   | 3  | 1  | 8  | 8   | 47.43GiB(74.13%) | 23  | 12.56  | 4.21  | 690      | [j7xpn4cr][j7xpn4cr] |
| ✅     | 80B_deep_alt | 24   | 6  | 1  | 4  | 4   | 41.63GiB(65.06%) | 41  | 22.43  | 7.52  | 245      | [okgm1auw][okgm1auw] |
| ✅     | 80B_deep_alt | 24   | 12 | 1  | 2  | 2   | 35.48GiB(55.45%) | 28  | 15.40  | 5.16  | 210      | [bea2k4u0][bea2k4u0] |

Logs:
`/flare/AuroraGPT/saforem2/projects/torchtitan-ezpz/outputs/benchmarks/80b_20260330_065453/`

## Sunspot Environment

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


| Status | Model        | TP | PP | DP | GBS | Memory           | TPS | TFLOPS | MFU   | Wall (s) | W&B                  |
|:------:|--------------|----|----|----|-----|------------------|-----|--------|-------|----------|:--------------------:|
| ✅     | 80B          | 2  | 1  | 12 | 12  | 59.82GiB(93.49%) | 85  | 46.31  | 15.53 | 331      | [qks5kewd][qks5kewd] |
| ✅     | 80B_alt      | 2  | 1  | 12 | 12  | 59.82GiB(93.49%) | 81  | 44.34  | 14.87 | 326      | [t60yimw5][t60yimw5] |
| ✅     | 80B_alt      | 3  | 1  | 8  | 8   | 51.61GiB(80.66%) | 23  | 12.40  | 4.16  | 694      | [mna9aesf][mna9aesf] |
| ✅     | 80B_alt      | 6  | 1  | 4  | 4   | 39.54GiB(61.79%) | 43  | 23.30  | 7.82  | 233      | [3j7ibvae][3j7ibvae] |
| ❌     | 80B_alt      | 12 | 1  | 2  | 2   | XXX              | XXX | XXX    | XXX   | 23       | NONE                 |
| ☠️     | 80B_wide     | 2  | 1  | 12 | 12  | 60.75GiB(94.94%) | 43  | 22.23  | 7.45  | 1002     | [i0fbuhab][i0fbuhab] |
| ✅     | 80B_wide     | 3  | 1  | 8  | 8   | 52.75GiB(82.44%) | 29  | 15.16  | 5.08  | 559      | [6vzikdog][6vzikdog] |
| ✅     | 80B_wide     | 6  | 1  | 4  | 4   | 40.47GiB(63.26%) | 51  | 26.35  | 8.84  | 216      | [ltaakcpx][ltaakcpx] |
| ❌     | 80B_wide     | 12 | 1  | 2  | 2   | XXX              | XXX | XXX    | XXX   | 22       | NONE                 |
| ✅     | 80B_deep     | 2  | 1  | 12 | 12  | 58.84GiB(91.95%) | 83  | 45.58  | 15.29 | 324      | [er9y8co7][er9y8co7] |
| ✅     | 80B_deep_alt | 2  | 1  | 12 | 12  | 58.95GiB(92.14%) | 77  | 41.87  | 14.04 | 343      | [924exg6x][924exg6x] |
| ✅     | 80B_deep_alt | 3  | 1  | 8  | 8   | 47.43GiB(74.13%) | 23  | 12.56  | 4.21  | 690      | [j7xpn4cr][j7xpn4cr] |
| ✅     | 80B_deep_alt | 6  | 1  | 4  | 4   | 41.63GiB(65.06%) | 41  | 22.43  | 7.52  | 245      | [okgm1auw][okgm1auw] |
| ✅     | 80B_deep_alt | 12 | 1  | 2  | 2   | 35.48GiB(55.45%) | 28  | 15.40  | 5.16  | 210      | [bea2k4u0][bea2k4u0] |


Logs:
`/tegu/datascience/foremans/projects/torchtitan/outputs/benchmarks/80b_20260330_065453/`

[qks5kewd]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/qks5kewd "qks5kewd"
[t60yimw5]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/t60yimw5 "t60yimw5"
[mna9aesf]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/mna9aesf "mna9aesf"
[3j7ibvae]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/3j7ibvae "3j7ibvae"
[i0fbuhab]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/i0fbuhab "i0fbuhab"
[6vzikdog]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/6vzikdog "6vzikdog"
[ltaakcpx]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ltaakcpx "ltaakcpx"
[er9y8co7]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/er9y8co7 "er9y8co7"
[924exg6x]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/924exg6x "924exg6x"
[j7xpn4cr]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/j7xpn4cr "j7xpn4cr"
[okgm1auw]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/okgm1auw "okgm1auw"
[bea2k4u0]: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/bea2k4u0 "bea2k4u0"
