# AML on the A2SL avocados: photo-only students taught by a photo + microwave teacher

Root: `results/a2sl_aml`. 4 folds (leave one of avocados 1, 2, 3, 11 out) x seeds; labels are training targets only. Every row except the teacher sees photos only at test time.

## fpt_filter, tracked avocados only (1, 2, 3, 11) (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.48 ± 0.24 | 1.12 ± 0.19 | 0.92 ± 0.10 | 0.81 ± 0.06 | — |
| CAML distillation (alpha 0.9, t 4) | 1.62 ± 0.06 | 1.21 ± 0.06 | 0.92 ± 0.04 | 0.77 ± 0.02 | +0.09 [-0.01, +0.27] |
| CAML distillation (alpha 0.5, t 1) | 1.56 ± 0.11 | 1.18 ± 0.10 | 0.92 ± 0.07 | 0.78 ± 0.06 | +0.06 [-0.01, +0.14] |
| feature matching (frozen teacher) | 1.60 ± 0.24 | 1.13 ± 0.15 | 0.96 ± 0.07 | 0.75 ± 0.11 | +0.00 [-0.04, +0.04] |
| CAML distillation + feature matching | 1.56 ± 0.06 | 1.17 ± 0.06 | 0.94 ± 0.06 | 0.80 ± 0.02 | +0.04 [-0.05, +0.17] |
| SAMD (teacher reset each round) | 1.98 ± 0.25 | 1.45 ± 0.18 | 0.85 ± 0.04 | 0.53 ± 0.15 | +0.19 [+0.08, +0.27] * |
| teacher: photo + microwave (needs the scanner) | 1.54 ± 0.04 | 1.18 ± 0.05 | 0.90 ± 0.04 | 0.79 ± 0.04 | +0.06 [-0.03, +0.19] |

## latent_sde, tracked avocados only (1, 2, 3, 11) (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 0.88 ± 0.23 | 0.63 ± 0.09 | 1.00 ± 0.00 | 0.99 ± 0.01 | — |
| CAML distillation (alpha 0.9, t 4) | 0.81 ± 0.23 | 0.64 ± 0.06 | 1.00 ± 0.00 | 0.98 ± 0.02 | +0.01 [-0.08, +0.09] |
| CAML distillation (alpha 0.5, t 1) | 0.94 ± 0.06 | 0.67 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.04 [-0.06, +0.13] |
| feature matching (frozen teacher) | 1.29 ± 0.32 | 0.79 ± 0.13 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.09 [-0.01, +0.18] |
| CAML distillation + feature matching | 0.94 ± 0.17 | 0.66 ± 0.11 | 1.00 ± 0.00 | 0.99 ± 0.01 | +0.00 [-0.03, +0.04] |
| SAMD (teacher reset each round) | 0.88 ± 0.23 | 0.63 ± 0.05 | 1.00 ± 0.00 | 0.99 ± 0.01 | -0.08 [-0.35, +0.19] |
| teacher: photo + microwave (needs the scanner) | 1.00 ± 0.23 | 0.71 ± 0.12 | 0.98 ± 0.04 | 0.97 ± 0.04 | +0.10 [+0.02, +0.17] * |

## fpt_filter, all test predictions (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 2.06 ± 0.23 | 1.62 ± 0.16 | 0.80 ± 0.08 | 0.66 ± 0.03 | — |
| CAML distillation (alpha 0.9, t 4) | 2.27 ± 0.07 | 1.74 ± 0.08 | 0.79 ± 0.04 | 0.62 ± 0.03 | +0.09 [+0.01, +0.18] * |
| CAML distillation (alpha 0.5, t 1) | 2.19 ± 0.17 | 1.67 ± 0.12 | 0.79 ± 0.07 | 0.61 ± 0.07 | +0.05 [+0.01, +0.09] * |
| feature matching (frozen teacher) | 2.15 ± 0.17 | 1.59 ± 0.09 | 0.84 ± 0.05 | 0.62 ± 0.05 | -0.01 [-0.04, +0.01] |
| CAML distillation + feature matching | 2.15 ± 0.09 | 1.66 ± 0.09 | 0.82 ± 0.05 | 0.64 ± 0.04 | +0.05 [-0.00, +0.11] |
| SAMD (teacher reset each round) | 2.40 ± 0.21 | 1.82 ± 0.16 | 0.76 ± 0.04 | 0.51 ± 0.08 | +0.08 [-0.10, +0.18] |
| teacher: photo + microwave (needs the scanner) | 2.10 ± 0.02 | 1.68 ± 0.08 | 0.79 ± 0.05 | 0.65 ± 0.01 | +0.04 [-0.01, +0.11] |

## latent_sde, all test predictions (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.62 ± 0.17 | 1.28 ± 0.06 | 0.88 ± 0.00 | 0.76 ± 0.01 | — |
| CAML distillation (alpha 0.9, t 4) | 1.59 ± 0.22 | 1.30 ± 0.07 | 0.88 ± 0.00 | 0.75 ± 0.02 | -0.01 [-0.06, +0.03] |
| CAML distillation (alpha 0.5, t 1) | 1.64 ± 0.07 | 1.29 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.00 | +0.01 [-0.04, +0.06] |
| feature matching (frozen teacher) | 1.85 ± 0.26 | 1.35 ± 0.09 | 0.88 ± 0.00 | 0.76 ± 0.00 | +0.11 [+0.02, +0.25] * |
| CAML distillation + feature matching | 1.62 ± 0.14 | 1.29 ± 0.09 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.01 [-0.05, +0.02] |
| SAMD (teacher reset each round) | 1.62 ± 0.14 | 1.28 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.00 | +0.12 [-0.17, +0.64] |
| teacher: photo + microwave (needs the scanner) | 1.73 ± 0.23 | 1.34 ± 0.11 | 0.86 ± 0.02 | 0.74 ± 0.02 | +0.08 [-0.00, +0.18] |

