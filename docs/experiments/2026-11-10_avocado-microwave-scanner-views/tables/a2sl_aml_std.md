# AML on the A2SL avocados: photo-only students taught by a photo + microwave teacher

Root: `results/a2sl_aml_std`. 4 folds (leave one of avocados 1, 2, 3, 11 out) x seeds; labels are training targets only. Every row except the teacher sees photos only at test time.

## latent_ode, tracked avocados only (1, 2, 3, 11) (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 0.44 ± 0.12 | 0.42 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | — |
| CAML distillation (alpha 0.9, t 4) | 0.58 ± 0.25 | 0.43 ± 0.12 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.04 [-0.07, +0.01] |
| SAMD (teacher reset each round) | 0.79 ± 0.19 | 0.47 ± 0.07 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.29 [-0.49, -0.08] * |
| lab method: reset + shared head + KD | 0.67 ± 0.04 | 0.40 ± 0.07 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.15 [-0.29, -0.02] * |
| reset + RKD | 0.85 ± 0.04 | 0.44 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.25 [-0.41, -0.03] * |
| reset + SP | 0.77 ± 0.14 | 0.47 ± 0.08 | 0.98 ± 0.04 | 0.99 ± 0.01 | -0.21 [-0.33, -0.09] * |
| control: reset schedule, no teaching term | 0.79 ± 0.16 | 0.46 ± 0.10 | 0.98 ± 0.04 | 0.99 ± 0.01 | -0.26 [-0.40, -0.11] * |
| control: reset + RKD, teacher sees photos only | 0.79 ± 0.07 | 0.44 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.23 [-0.40, -0.02] * |
| control: SAMD, teacher sees photos only | 0.79 ± 0.19 | 0.47 ± 0.07 | 0.98 ± 0.04 | 1.00 ± 0.00 | -0.29 [-0.49, -0.08] * |
| SAMD, teacher terms on scanned fruit only | 0.62 ± 0.38 | 0.41 ± 0.12 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.19 [-0.41, +0.03] |
| reset + RKD, teacher terms on scanned fruit only | 0.56 ± 0.22 | 0.42 ± 0.06 | 1.00 ± 0.00 | 0.99 ± 0.01 | -0.13 [-0.29, +0.07] |
| teacher: photo + microwave (needs the scanner) | 0.56 ± 0.23 | 0.40 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.02 [-0.25, +0.20] |

## latent_sde, tracked avocados only (1, 2, 3, 11) (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 0.92 ± 0.19 | 0.61 ± 0.13 | 1.00 ± 0.00 | 0.99 ± 0.01 | — |
| CAML distillation (alpha 0.9, t 4) | 0.92 ± 0.34 | 0.69 ± 0.14 | 1.00 ± 0.00 | 0.97 ± 0.03 | +0.09 [-0.10, +0.26] |
| CAML distillation (alpha 0.5, t 1) | 1.06 ± 0.12 | 0.69 ± 0.08 | 1.00 ± 0.00 | 0.99 ± 0.01 | +0.08 [+0.01, +0.15] * |
| feature matching (frozen teacher) | 1.02 ± 0.13 | 0.68 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.07 [-0.07, +0.24] |
| CAML distillation + feature matching | 0.79 ± 0.20 | 0.64 ± 0.06 | 1.00 ± 0.00 | 0.99 ± 0.01 | +0.03 [-0.10, +0.15] |
| relational KD (RKD, frozen teacher) | 1.10 ± 0.07 | 0.73 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.01 | +0.06 [-0.04, +0.18] |
| probabilistic KT (PKT, frozen teacher) | 0.94 ± 0.17 | 0.67 ± 0.05 | 1.00 ± 0.00 | 0.99 ± 0.01 | +0.05 [-0.02, +0.13] |
| similarity preserving (SP, frozen teacher) | 1.17 ± 0.30 | 0.75 ± 0.07 | 1.00 ± 0.00 | 0.99 ± 0.01 | +0.05 [-0.06, +0.18] |
| SAMD (teacher reset each round) | 1.00 ± 0.23 | 0.62 ± 0.06 | 1.00 ± 0.00 | 0.99 ± 0.00 | -0.10 [-0.36, +0.15] |
| lab method: reset + shared head + KD | 0.90 ± 0.24 | 0.60 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.01 | -0.10 [-0.20, +0.01] |
| reset + RKD | 0.77 ± 0.14 | 0.50 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.33 [-0.48, -0.18] * |
| reset + PKT | 1.35 ± 0.14 | 0.76 ± 0.06 | 0.98 ± 0.04 | 0.99 ± 0.01 | -0.01 [-0.27, +0.25] |
| reset + SP | 0.77 ± 0.22 | 0.49 ± 0.08 | 1.00 ± 0.00 | 1.00 ± 0.01 | -0.30 [-0.50, -0.10] * |
| control: reset schedule, no teaching term | 0.73 ± 0.14 | 0.42 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.40 [-0.54, -0.26] * |
| control: reset + RKD, teacher sees photos only | 0.83 ± 0.30 | 0.61 ± 0.08 | 1.00 ± 0.00 | 0.99 ± 0.02 | -0.08 [-0.28, +0.13] |
| control: SAMD, teacher sees photos only | 0.75 ± 0.17 | 0.50 ± 0.05 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.28 [-0.44, -0.12] * |
| CAML distillation, teacher terms on scanned fruit only | 0.65 ± 0.10 | 0.66 ± 0.04 | 1.00 ± 0.00 | 0.98 ± 0.00 | +0.19 [+0.02, +0.35] * |
| SAMD, teacher terms on scanned fruit only | 0.79 ± 0.10 | 0.46 ± 0.07 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.34 [-0.54, -0.19] * |
| lab method, teacher terms on scanned fruit only | 0.81 ± 0.22 | 0.63 ± 0.03 | 1.00 ± 0.00 | 0.98 ± 0.03 | +0.03 [-0.16, +0.18] |
| teacher: photo + microwave (needs the scanner) | 0.96 ± 0.16 | 0.68 ± 0.08 | 1.00 ± 0.00 | 0.99 ± 0.02 | +0.07 [+0.01, +0.14] * |
| ours_rkd_b01_lr1e3 | 0.81 ± 0.11 | 0.50 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.38 [-0.50, -0.26] * |
| ours_rkd_b01_lr3e3 | 0.67 ± 0.38 | 0.46 ± 0.14 | 0.94 ± 0.06 | 0.99 ± 0.01 | -0.42 [-0.48, -0.35] * |
| ours_rkd_b01_lr3e4 | 0.88 ± 0.11 | 0.63 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.05 [-0.09, +0.19] |
| ours_rkd_b10_lr1e3 | 0.54 ± 0.07 | 0.42 ± 0.05 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.36 [-0.46, -0.20] * |
| ours_rkd_b10_lr3e3 | 0.71 ± 0.26 | 0.50 ± 0.10 | 1.00 ± 0.00 | 0.98 ± 0.03 | -0.31 [-0.43, -0.18] * |
| ours_rkd_b10_lr3e4 | 0.81 ± 0.06 | 0.64 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.08 [-0.05, +0.21] |
| ours_rkd_b1_lr1e3 | 0.56 ± 0.06 | 0.45 ± 0.07 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.37 [-0.44, -0.29] * |
| ours_rkd_b1_lr3e3 | 0.71 ± 0.22 | 0.41 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.51 [-0.62, -0.39] * |
| ours_rkd_b1_lr3e4 | 0.83 ± 0.10 | 0.63 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.01 | +0.08 [-0.12, +0.27] |
| samd_paired_b01_lr1e3 | 0.85 ± 0.13 | 0.55 ± 0.08 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.18 [-0.28, -0.05] * |
| samd_paired_b01_lr3e3 | 0.56 ± 0.06 | 0.50 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.37 [-0.45, -0.29] * |
| samd_paired_b01_lr3e4 | 0.83 ± 0.30 | 0.65 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.12 [+0.02, +0.25] * |
| samd_paired_b10_lr1e3 | 0.94 ± 0.12 | 0.63 ± 0.05 | 1.00 ± 0.00 | 0.99 ± 0.01 | -0.03 [-0.13, +0.07] |
| samd_paired_b10_lr3e3 | 0.85 ± 0.13 | 0.47 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.45 [-0.60, -0.30] * |
| samd_paired_b10_lr3e4 | 0.81 ± 0.23 | 0.65 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.08 [-0.02, +0.24] |
| samd_paired_b1_lr1e3 | 0.90 ± 0.20 | 0.57 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.14 [-0.21, -0.01] * |
| samd_paired_b1_lr3e3 | 0.90 ± 0.10 | 0.55 ± 0.02 | 1.00 ± 0.00 | 0.99 ± 0.01 | -0.35 [-0.50, -0.19] * |
| samd_paired_b1_lr3e4 | 0.83 ± 0.25 | 0.64 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.09 [-0.00, +0.24] |

## latent_sde, tracked avocados only (1, 2, 3, 11) (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 0.90 ± 0.20 | 0.64 ± 0.05 | 1.00 ± 0.00 | 0.98 ± 0.02 | — |
| CAML distillation (alpha 0.9, t 4) | 0.83 ± 0.10 | 0.67 ± 0.09 | 1.00 ± 0.00 | 0.98 ± 0.04 | +0.08 [-0.06, +0.19] |
| CAML distillation (alpha 0.5, t 1) | 1.02 ± 0.07 | 0.66 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | -0.00 [-0.04, +0.05] |
| feature matching (frozen teacher) | 1.00 ± 0.11 | 0.68 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | +0.02 [-0.10, +0.15] |
| CAML distillation + feature matching | 0.85 ± 0.30 | 0.64 ± 0.14 | 1.00 ± 0.00 | 0.98 ± 0.04 | +0.01 [-0.06, +0.06] |
| SAMD (teacher reset each round) | 1.04 ± 0.24 | 0.65 ± 0.05 | 1.00 ± 0.00 | 0.99 ± 0.00 | -0.17 [-0.46, +0.11] |
| teacher: photo + microwave (needs the scanner) | 1.04 ± 0.19 | 0.68 ± 0.12 | 1.00 ± 0.00 | 0.99 ± 0.02 | +0.03 [-0.06, +0.11] |

## latent_ode, all test predictions (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.27 ± 0.05 | 1.18 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.00 | — |
| CAML distillation (alpha 0.9, t 4) | 1.39 ± 0.27 | 1.20 ± 0.09 | 0.88 ± 0.00 | 0.77 ± 0.00 | -0.03 [-0.06, -0.01] * |
| SAMD (teacher reset each round) | 1.53 ± 0.08 | 1.21 ± 0.03 | 0.88 ± 0.00 | 0.77 ± 0.01 | -0.23 [-0.35, -0.10] * |
| lab method: reset + shared head + KD | 1.51 ± 0.04 | 1.17 ± 0.07 | 0.88 ± 0.00 | 0.77 ± 0.01 | -0.13 [-0.21, -0.05] * |
| reset + RKD | 1.54 ± 0.08 | 1.17 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.19 [-0.30, -0.06] * |
| reset + SP | 1.51 ± 0.16 | 1.20 ± 0.08 | 0.86 ± 0.02 | 0.76 ± 0.00 | -0.18 [-0.25, -0.09] * |
| control: reset schedule, no teaching term | 1.52 ± 0.21 | 1.20 ± 0.09 | 0.85 ± 0.04 | 0.76 ± 0.00 | -0.22 [-0.32, -0.10] * |
| control: reset + RKD, teacher sees photos only | 1.50 ± 0.08 | 1.17 ± 0.02 | 0.88 ± 0.00 | 0.77 ± 0.01 | -0.18 [-0.29, -0.05] * |
| control: SAMD, teacher sees photos only | 1.53 ± 0.08 | 1.20 ± 0.03 | 0.86 ± 0.02 | 0.77 ± 0.01 | -0.23 [-0.35, -0.10] * |
| SAMD, teacher terms on scanned fruit only | 1.47 ± 0.34 | 1.18 ± 0.11 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.16 [-0.29, -0.03] * |
| reset + RKD, teacher terms on scanned fruit only | 1.40 ± 0.15 | 1.19 ± 0.06 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.10 [-0.19, +0.00] |
| teacher: photo + microwave (needs the scanner) | 1.36 ± 0.18 | 1.16 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.04 [-0.17, +0.09] |

## latent_sde, all test predictions (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.64 ± 0.13 | 1.28 ± 0.08 | 0.88 ± 0.00 | 0.76 ± 0.00 | — |
| CAML distillation (alpha 0.9, t 4) | 1.58 ± 0.25 | 1.33 ± 0.12 | 0.88 ± 0.00 | 0.74 ± 0.02 | +0.09 [-0.01, +0.18] |
| CAML distillation (alpha 0.5, t 1) | 1.65 ± 0.13 | 1.30 ± 0.06 | 0.88 ± 0.00 | 0.76 ± 0.00 | +0.06 [+0.00, +0.11] * |
| feature matching (frozen teacher) | 1.66 ± 0.11 | 1.32 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.00 | +0.14 [-0.01, +0.40] |
| CAML distillation + feature matching | 1.51 ± 0.17 | 1.30 ± 0.07 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.02 [-0.05, +0.09] |
| relational KD (RKD, frozen teacher) | 1.74 ± 0.07 | 1.34 ± 0.01 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.11 [+0.00, +0.29] * |
| probabilistic KT (PKT, frozen teacher) | 1.62 ± 0.11 | 1.32 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.02 [-0.02, +0.06] |
| similarity preserving (SP, frozen teacher) | 1.80 ± 0.17 | 1.34 ± 0.05 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.12 [-0.00, +0.33] |
| SAMD (teacher reset each round) | 1.73 ± 0.12 | 1.25 ± 0.01 | 0.88 ± 0.00 | 0.75 ± 0.01 | +0.05 [-0.19, +0.45] |
| lab method: reset + shared head + KD | 1.59 ± 0.11 | 1.25 ± 0.03 | 0.88 ± 0.00 | 0.75 ± 0.02 | -0.08 [-0.14, -0.01] * |
| reset + RKD | 1.50 ± 0.08 | 1.21 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.22 [-0.32, -0.11] * |
| reset + PKT | 1.93 ± 0.13 | 1.35 ± 0.09 | 0.86 ± 0.02 | 0.76 ± 0.01 | +0.11 [-0.12, +0.46] |
| reset + SP | 1.52 ± 0.18 | 1.21 ± 0.05 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.20 [-0.32, -0.08] * |
| control: reset schedule, no teaching term | 1.50 ± 0.14 | 1.14 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.26 [-0.39, -0.10] * |
| control: reset + RKD, teacher sees photos only | 1.56 ± 0.16 | 1.26 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.00 | +0.06 [-0.15, +0.44] |
| control: SAMD, teacher sees photos only | 1.48 ± 0.13 | 1.20 ± 0.03 | 0.88 ± 0.00 | 0.77 ± 0.01 | -0.17 [-0.29, -0.03] * |
| CAML distillation, teacher terms on scanned fruit only | 1.38 ± 0.14 | 1.32 ± 0.02 | 0.88 ± 0.00 | 0.75 ± 0.01 | +0.19 [+0.08, +0.32] * |
| SAMD, teacher terms on scanned fruit only | 1.54 ± 0.04 | 1.17 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.25 [-0.37, -0.12] * |
| lab method, teacher terms on scanned fruit only | 1.53 ± 0.24 | 1.29 ± 0.05 | 0.88 ± 0.00 | 0.75 ± 0.01 | +0.05 [-0.06, +0.16] |
| teacher: photo + microwave (needs the scanner) | 1.70 ± 0.15 | 1.32 ± 0.09 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.09 [+0.01, +0.22] * |
| ours_rkd_b01_lr1e3 | 1.56 ± 0.08 | 1.20 ± 0.06 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.27 [-0.37, -0.15] * |
| ours_rkd_b01_lr3e3 | 1.45 ± 0.31 | 1.18 ± 0.08 | 0.82 ± 0.07 | 0.76 ± 0.00 | -0.28 [-0.37, -0.15] * |
| ours_rkd_b01_lr3e4 | 1.56 ± 0.06 | 1.27 ± 0.05 | 0.88 ± 0.00 | 0.77 ± 0.01 | +0.02 [-0.06, +0.09] |
| ours_rkd_b10_lr1e3 | 1.32 ± 0.08 | 1.16 ± 0.05 | 0.88 ± 0.00 | 0.77 ± 0.01 | -0.24 [-0.34, -0.11] * |
| ours_rkd_b10_lr3e3 | 1.53 ± 0.20 | 1.22 ± 0.08 | 0.88 ± 0.00 | 0.75 ± 0.01 | -0.19 [-0.29, -0.07] * |
| ours_rkd_b10_lr3e4 | 1.56 ± 0.06 | 1.29 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.04 [-0.04, +0.12] |
| ours_rkd_b1_lr1e3 | 1.32 ± 0.11 | 1.16 ± 0.06 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.24 [-0.34, -0.12] * |
| ours_rkd_b1_lr3e3 | 1.48 ± 0.19 | 1.15 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.35 [-0.47, -0.18] * |
| ours_rkd_b1_lr3e4 | 1.56 ± 0.09 | 1.28 ± 0.04 | 0.88 ± 0.00 | 0.77 ± 0.01 | +0.02 [-0.08, +0.12] |
| samd_paired_b01_lr1e3 | 1.59 ± 0.09 | 1.24 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.00 [-0.19, +0.37] |
| samd_paired_b01_lr3e3 | 1.38 ± 0.14 | 1.22 ± 0.06 | 0.86 ± 0.02 | 0.76 ± 0.01 | -0.08 [-0.33, +0.45] |
| samd_paired_b01_lr3e4 | 1.56 ± 0.26 | 1.32 ± 0.04 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.13 [+0.04, +0.23] * |
| samd_paired_b10_lr1e3 | 1.66 ± 0.17 | 1.29 ± 0.06 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.08 [-0.07, +0.37] |
| samd_paired_b10_lr3e3 | 1.57 ± 0.08 | 1.17 ± 0.01 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.13 [-0.41, +0.42] |
| samd_paired_b10_lr3e4 | 1.56 ± 0.26 | 1.33 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.17 [+0.02, +0.43] * |
| samd_paired_b1_lr1e3 | 1.61 ± 0.15 | 1.24 ± 0.05 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.05 [-0.15, +0.47] |
| samd_paired_b1_lr3e3 | 1.60 ± 0.15 | 1.22 ± 0.05 | 0.88 ± 0.00 | 0.76 ± 0.01 | -0.07 [-0.34, +0.46] |
| samd_paired_b1_lr3e4 | 1.57 ± 0.26 | 1.33 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.16 [+0.03, +0.39] * |

## latent_sde, all test predictions (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.61 ± 0.12 | 1.31 ± 0.03 | 0.88 ± 0.00 | 0.76 ± 0.00 | — |
| CAML distillation (alpha 0.9, t 4) | 1.59 ± 0.11 | 1.33 ± 0.10 | 0.88 ± 0.00 | 0.75 ± 0.02 | +0.04 [-0.04, +0.11] |
| CAML distillation (alpha 0.5, t 1) | 1.68 ± 0.04 | 1.30 ± 0.01 | 0.88 ± 0.00 | 0.76 ± 0.00 | -0.00 [-0.07, +0.05] |
| feature matching (frozen teacher) | 1.66 ± 0.11 | 1.30 ± 0.01 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.10 [-0.02, +0.27] |
| CAML distillation + feature matching | 1.56 ± 0.24 | 1.28 ± 0.09 | 0.88 ± 0.00 | 0.75 ± 0.02 | -0.01 [-0.07, +0.04] |
| SAMD (teacher reset each round) | 1.73 ± 0.11 | 1.27 ± 0.01 | 0.88 ± 0.00 | 0.76 ± 0.01 | +0.06 [-0.23, +0.57] |
| teacher: photo + microwave (needs the scanner) | 1.70 ± 0.23 | 1.32 ± 0.11 | 0.88 ± 0.00 | 0.75 ± 0.01 | +0.08 [+0.01, +0.20] * |

