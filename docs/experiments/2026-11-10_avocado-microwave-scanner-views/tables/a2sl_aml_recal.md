# AML on the A2SL avocados: photo-only students taught by a photo + microwave teacher, recalibrated

Root: `results/a2sl_aml`. 4 folds (leave one of avocados 1, 2, 3, 11 out) x seeds; labels are training targets only. Every row except the teacher sees photos only at test time.

## fpt_filter, tracked avocados only (1, 2, 3, 11) (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.44 ± 0.12 | 1.00 ± 0.08 | 0.90 ± 0.04 | 0.82 ± 0.08 | — |
| CAML distillation (alpha 0.9, t 4) | 1.67 ± 0.04 | 1.07 ± 0.03 | 0.94 ± 0.00 | 0.69 ± 0.02 | +0.16 [+0.05, +0.28] * |
| CAML distillation (alpha 0.5, t 1) | 1.58 ± 0.16 | 1.06 ± 0.06 | 0.90 ± 0.04 | 0.74 ± 0.07 | +0.12 [+0.04, +0.20] * |
| feature matching (frozen teacher) | 1.52 ± 0.10 | 1.05 ± 0.12 | 0.88 ± 0.00 | 0.82 ± 0.03 | +0.10 [+0.06, +0.13] * |
| CAML distillation + feature matching | 1.60 ± 0.10 | 1.07 ± 0.01 | 0.92 ± 0.04 | 0.72 ± 0.05 | +0.17 [+0.04, +0.30] * |
| SAMD (teacher reset each round) | 1.88 ± 0.32 | 1.23 ± 0.11 | 0.85 ± 0.07 | 0.60 ± 0.15 | +0.22 [+0.10, +0.29] * |
| teacher: photo + microwave (needs the scanner) | 1.54 ± 0.04 | 1.05 ± 0.03 | 0.92 ± 0.04 | 0.80 ± 0.02 | +0.05 [+0.00, +0.11] * |

## latent_sde, tracked avocados only (1, 2, 3, 11) (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 0.77 ± 0.18 | 0.66 ± 0.10 | 0.71 ± 0.04 | 0.98 ± 0.02 | — |
| CAML distillation (alpha 0.9, t 4) | 0.79 ± 0.04 | 0.68 ± 0.12 | 0.75 ± 0.00 | 0.98 ± 0.01 | +0.27 [-0.00, +0.70] |
| CAML distillation (alpha 0.5, t 1) | 0.77 ± 0.13 | 0.62 ± 0.04 | 0.67 ± 0.04 | 1.00 ± 0.01 | -0.13 [-0.38, +0.01] |
| feature matching (frozen teacher) | 0.81 ± 0.12 | 0.68 ± 0.09 | 0.75 ± 0.11 | 0.99 ± 0.01 | -0.13 [-0.24, -0.02] * |
| CAML distillation + feature matching | 0.83 ± 0.25 | 0.69 ± 0.17 | 0.77 ± 0.07 | 0.98 ± 0.03 | +0.14 [-0.22, +0.46] |
| SAMD (teacher reset each round) | 0.88 ± 0.11 | 0.62 ± 0.05 | 0.77 ± 0.07 | 0.97 ± 0.03 | +0.03 [-0.18, +0.24] |
| teacher: photo + microwave (needs the scanner) | 1.04 ± 0.34 | 0.76 ± 0.21 | 0.71 ± 0.04 | 0.96 ± 0.04 | +0.33 [+0.06, +0.55] * |

## fpt_filter, all test predictions (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.91 ± 0.08 | 1.35 ± 0.04 | 0.79 ± 0.05 | 0.64 ± 0.05 | — |
| CAML distillation (alpha 0.9, t 4) | 2.10 ± 0.07 | 1.42 ± 0.04 | 0.81 ± 0.00 | 0.57 ± 0.01 | +0.19 [+0.07, +0.36] * |
| CAML distillation (alpha 0.5, t 1) | 2.01 ± 0.12 | 1.39 ± 0.03 | 0.79 ± 0.04 | 0.59 ± 0.04 | +0.14 [+0.05, +0.27] * |
| feature matching (frozen teacher) | 1.95 ± 0.05 | 1.39 ± 0.07 | 0.76 ± 0.02 | 0.64 ± 0.01 | +0.09 [+0.01, +0.21] * |
| CAML distillation + feature matching | 2.02 ± 0.10 | 1.41 ± 0.02 | 0.79 ± 0.04 | 0.59 ± 0.04 | +0.16 [+0.06, +0.26] * |
| SAMD (teacher reset each round) | 2.15 ± 0.24 | 1.47 ± 0.08 | 0.78 ± 0.05 | 0.52 ± 0.09 | +0.12 [-0.02, +0.20] |
| teacher: photo + microwave (needs the scanner) | 2.02 ± 0.04 | 1.38 ± 0.02 | 0.79 ± 0.04 | 0.62 ± 0.03 | +0.07 [+0.02, +0.12] * |

## latent_sde, all test predictions (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.51 ± 0.09 | 1.23 ± 0.07 | 0.66 ± 0.06 | 0.75 ± 0.01 | — |
| CAML distillation (alpha 0.9, t 4) | 1.59 ± 0.06 | 1.30 ± 0.13 | 0.64 ± 0.02 | 0.75 ± 0.01 | +0.20 [-0.01, +0.46] |
| CAML distillation (alpha 0.5, t 1) | 1.50 ± 0.03 | 1.22 ± 0.02 | 0.60 ± 0.02 | 0.76 ± 0.01 | -0.12 [-0.23, -0.03] * |
| feature matching (frozen teacher) | 1.54 ± 0.13 | 1.23 ± 0.09 | 0.69 ± 0.11 | 0.76 ± 0.00 | -0.04 [-0.14, +0.12] |
| CAML distillation + feature matching | 1.57 ± 0.21 | 1.28 ± 0.15 | 0.66 ± 0.03 | 0.75 ± 0.02 | +0.07 [-0.10, +0.25] |
| SAMD (teacher reset each round) | 1.62 ± 0.05 | 1.24 ± 0.05 | 0.67 ± 0.07 | 0.74 ± 0.01 | +0.12 [-0.12, +0.57] |
| teacher: photo + microwave (needs the scanner) | 1.76 ± 0.32 | 1.35 ± 0.20 | 0.67 ± 0.02 | 0.74 ± 0.02 | +0.21 [+0.07, +0.36] * |

