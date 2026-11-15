# AML on the A2SL avocados: photo-only students taught by a photo + microwave teacher

Root: `outputs/a2sl_schedule`. 4 folds (leave one of avocados 1, 2, 3, 11 out) x seeds; labels are training targets only. Every row except the teacher sees photos only at test time.

## latent_ode, tracked avocados only (1, 2, 3, 11) (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 0.40 ± 0.15 | 0.42 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | — |
| CAML distillation (alpha 0.9, t 4) | 0.54 ± 0.19 | 0.42 ± 0.08 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.04 [-0.10, +0.03] |
| SAMD (teacher reset each round) | 0.72 ± 0.17 | 0.45 ± 0.06 | 0.99 ± 0.03 | 1.00 ± 0.00 | -0.31 [-0.44, -0.17] * |
| lab method: reset + shared predictor + KD | 0.65 ± 0.14 | 0.40 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.23 [-0.39, -0.11] * |
| reset + RKD | 0.72 ± 0.19 | 0.44 ± 0.04 | 0.99 ± 0.03 | 1.00 ± 0.00 | -0.23 [-0.40, -0.10] * |
| control: photo only, stopped on all-visits validation | 0.46 ± 0.22 | 0.40 ± 0.05 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.11 [-0.32, +0.07] |
| control: reset rounds, no teacher phase | 0.53 ± 0.19 | 0.40 ± 0.08 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.09 [-0.26, +0.02] |
| control: reset schedule, photo-only teacher, no teaching term | 0.69 ± 0.19 | 0.46 ± 0.08 | 0.97 ± 0.03 | 1.00 ± 0.01 | -0.26 [-0.33, -0.19] * |
| control: reset schedule, no teaching term | 0.68 ± 0.19 | 0.44 ± 0.08 | 0.97 ± 0.03 | 1.00 ± 0.01 | -0.27 [-0.35, -0.19] * |
| control: SAMD, teacher sees photos only | 0.72 ± 0.17 | 0.45 ± 0.06 | 0.97 ± 0.03 | 1.00 ± 0.00 | -0.31 [-0.44, -0.17] * |
| teacher: photo + microwave (needs the scanner) | 0.49 ± 0.19 | 0.40 ± 0.05 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.05 [-0.26, +0.11] |

## latent_sde, tracked avocados only (1, 2, 3, 11) (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 0.82 ± 0.17 | 0.63 ± 0.05 | 1.00 ± 0.00 | 1.00 ± 0.00 | — |
| CAML distillation (alpha 0.9, t 4) | 0.99 ± 0.26 | 0.71 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.13 [+0.03, +0.22] * |
| SAMD (teacher reset each round) | 1.18 ± 0.22 | 0.68 ± 0.10 | 1.00 ± 0.00 | 0.98 ± 0.03 | -0.08 [-0.18, +0.01] |
| lab method: reset + shared predictor + KD | 1.02 ± 0.19 | 0.67 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.05 [-0.06, +0.18] |
| reset + RKD | 1.12 ± 0.10 | 0.69 ± 0.04 | 1.00 ± 0.00 | 0.99 ± 0.01 | -0.10 [-0.17, -0.02] * |
| control: photo only, stopped on all-visits validation | 1.01 ± 0.28 | 0.68 ± 0.14 | 1.00 ± 0.00 | 1.00 ± 0.01 | +0.02 [-0.07, +0.11] |
| control: reset rounds, no teacher phase | 0.99 ± 0.09 | 0.63 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.03 [-0.11, +0.05] |
| control: reset schedule, photo-only teacher, no teaching term | 1.07 ± 0.20 | 0.69 ± 0.06 | 1.00 ± 0.00 | 0.99 ± 0.01 | -0.05 [-0.13, +0.04] |
| control: reset schedule, no teaching term | 1.14 ± 0.05 | 0.68 ± 0.04 | 1.00 ± 0.00 | 0.99 ± 0.01 | -0.06 [-0.18, +0.02] |
| control: SAMD, teacher sees photos only | 1.14 ± 0.12 | 0.65 ± 0.04 | 1.00 ± 0.00 | 0.99 ± 0.01 | -0.11 [-0.22, -0.01] * |
| teacher: photo + microwave (needs the scanner) | 0.96 ± 0.13 | 0.70 ± 0.08 | 1.00 ± 0.00 | 0.99 ± 0.01 | +0.11 [+0.02, +0.20] * |

## latent_ode, all test predictions (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.26 ± 0.07 | 1.18 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.00 | — |
| CAML distillation (alpha 0.9, t 4) | 1.35 ± 0.19 | 1.20 ± 0.06 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.02 [-0.05, +0.16] |
| SAMD (teacher reset each round) | 1.49 ± 0.08 | 1.19 ± 0.03 | 0.87 ± 0.01 | 0.76 ± 0.01 | -0.19 [-0.31, -0.01] * |
| lab method: reset + shared predictor + KD | 1.49 ± 0.11 | 1.17 ± 0.05 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.13 [-0.25, +0.04] |
| reset + RKD | 1.47 ± 0.14 | 1.18 ± 0.04 | 0.87 ± 0.01 | 0.76 ± 0.00 | -0.17 [-0.28, -0.07] * |
| control: photo only, stopped on all-visits validation | 1.32 ± 0.20 | 1.17 ± 0.05 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.07 [-0.17, +0.02] |
| control: reset rounds, no teacher phase | 1.38 ± 0.15 | 1.18 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.09 [-0.17, -0.01] * |
| control: reset schedule, photo-only teacher, no teaching term | 1.47 ± 0.16 | 1.20 ± 0.07 | 0.86 ± 0.03 | 0.76 ± 0.01 | -0.16 [-0.26, +0.01] |
| control: reset schedule, no teaching term | 1.45 ± 0.18 | 1.19 ± 0.07 | 0.86 ± 0.03 | 0.76 ± 0.01 | -0.16 [-0.27, +0.01] |
| control: SAMD, teacher sees photos only | 1.49 ± 0.08 | 1.19 ± 0.03 | 0.86 ± 0.02 | 0.76 ± 0.01 | -0.19 [-0.31, -0.01] * |
| teacher: photo + microwave (needs the scanner) | 1.31 ± 0.14 | 1.16 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.01 [-0.14, +0.15] |

## latent_sde, all test predictions (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.57 ± 0.14 | 1.29 ± 0.02 | 0.88 ± 0.00 | 0.76 ± 0.01 | — |
| CAML distillation (alpha 0.9, t 4) | 1.73 ± 0.24 | 1.32 ± 0.05 | 0.88 ± 0.00 | 0.75 ± 0.01 | +0.01 [-0.18, +0.12] |
| SAMD (teacher reset each round) | 1.81 ± 0.09 | 1.31 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.06 [-0.12, +0.00] |
| lab method: reset + shared predictor + KD | 1.71 ± 0.14 | 1.30 ± 0.05 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.04 [-0.21, +0.06] |
| reset + RKD | 1.84 ± 0.07 | 1.32 ± 0.01 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.02 [-0.10, +0.14] |
| control: photo only, stopped on all-visits validation | 1.73 ± 0.24 | 1.32 ± 0.10 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.03 [-0.13, +0.04] |
| control: reset rounds, no teacher phase | 1.70 ± 0.08 | 1.29 ± 0.05 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.06 [-0.13, -0.00] * |
| control: reset schedule, photo-only teacher, no teaching term | 1.80 ± 0.19 | 1.32 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.05 [-0.09, +0.00] |
| control: reset schedule, no teaching term | 1.82 ± 0.08 | 1.32 ± 0.02 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.05 [-0.11, -0.00] * |
| control: SAMD, teacher sees photos only | 1.81 ± 0.07 | 1.30 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.04 [-0.14, +0.13] |
| teacher: photo + microwave (needs the scanner) | 1.69 ± 0.13 | 1.33 ± 0.06 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.00 [-0.19, +0.10] |

