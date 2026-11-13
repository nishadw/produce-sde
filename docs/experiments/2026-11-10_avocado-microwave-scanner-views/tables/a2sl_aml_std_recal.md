# AML on the A2SL avocados: photo-only students taught by a photo + microwave teacher, recalibrated

Root: `results/a2sl_aml_std`. 4 folds (leave one of avocados 1, 2, 3, 11 out) x seeds; labels are training targets only. Every row except the teacher sees photos only at test time.

## latent_ode, tracked avocados only (1, 2, 3, 11) (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 0.56 ± 0.12 | 0.50 ± 0.06 | 0.75 ± 0.00 | 1.00 ± 0.00 | — |
| CAML distillation (alpha 0.9, t 4) | 0.54 ± 0.10 | 0.47 ± 0.05 | 0.75 ± 0.06 | 1.00 ± 0.00 | +0.02 [-0.08, +0.13] |
| SAMD (teacher reset each round) | 0.75 ± 0.17 | 0.65 ± 0.23 | 0.54 ± 0.36 | 1.00 ± 0.01 | +0.05 [-0.05, +0.15] |
| lab method: reset + shared head + KD | 0.69 ± 0.17 | 0.54 ± 0.07 | 0.73 ± 0.14 | 1.00 ± 0.00 | +0.09 [-0.00, +0.19] |
| reset + RKD | 0.77 ± 0.04 | 0.59 ± 0.08 | 0.62 ± 0.12 | 1.00 ± 0.00 | -0.05 [-0.10, -0.01] * |
| reset + SP | 0.67 ± 0.16 | 0.60 ± 0.07 | 0.65 ± 0.18 | 1.00 ± 0.01 | +0.09 [+0.00, +0.17] * |
| control: reset schedule, no teaching term | 0.69 ± 0.17 | 0.61 ± 0.04 | 0.62 ± 0.17 | 0.99 ± 0.01 | +0.12 [-0.02, +0.30] |
| control: reset + RKD, teacher sees photos only | 0.73 ± 0.10 | 0.60 ± 0.11 | 0.60 ± 0.13 | 1.00 ± 0.00 | -0.05 [-0.10, -0.00] * |
| control: SAMD, teacher sees photos only | 0.77 ± 0.16 | 0.65 ± 0.23 | 0.54 ± 0.36 | 1.00 ± 0.01 | +0.05 [-0.06, +0.15] |
| SAMD, teacher terms on scanned fruit only | 0.67 ± 0.18 | 0.54 ± 0.12 | 0.71 ± 0.10 | 1.00 ± 0.00 | -0.00 [-0.10, +0.16] |
| reset + RKD, teacher terms on scanned fruit only | 0.73 ± 0.13 | 0.59 ± 0.12 | 0.65 ± 0.16 | 0.99 ± 0.02 | +0.07 [-0.09, +0.30] |
| teacher: photo + microwave (needs the scanner) | 0.65 ± 0.14 | 0.51 ± 0.10 | 0.81 ± 0.11 | 1.00 ± 0.00 | -0.01 [-0.06, +0.04] |

## latent_sde, tracked avocados only (1, 2, 3, 11) (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 0.81 ± 0.17 | 0.64 ± 0.12 | 0.77 ± 0.10 | 0.98 ± 0.02 | — |
| CAML distillation (alpha 0.9, t 4) | 1.04 ± 0.38 | 0.80 ± 0.21 | 0.69 ± 0.06 | 0.95 ± 0.04 | +0.32 [-0.17, +0.77] |
| CAML distillation (alpha 0.5, t 1) | 0.90 ± 0.16 | 0.68 ± 0.09 | 0.81 ± 0.06 | 0.98 ± 0.01 | +0.12 [-0.01, +0.31] |
| feature matching (frozen teacher) | 0.81 ± 0.17 | 0.65 ± 0.09 | 0.69 ± 0.06 | 1.00 ± 0.01 | -0.07 [-0.33, +0.19] |
| CAML distillation + feature matching | 0.83 ± 0.20 | 0.70 ± 0.18 | 0.65 ± 0.16 | 0.97 ± 0.03 | +0.10 [-0.27, +0.47] |
| relational KD (RKD, frozen teacher) | 0.81 ± 0.06 | 0.69 ± 0.02 | 0.69 ± 0.06 | 0.99 ± 0.00 | +0.04 [-0.03, +0.11] |
| probabilistic KT (PKT, frozen teacher) | 0.75 ± 0.11 | 0.67 ± 0.10 | 0.69 ± 0.11 | 0.98 ± 0.02 | +0.05 [-0.35, +0.43] |
| similarity preserving (SP, frozen teacher) | 0.83 ± 0.26 | 0.70 ± 0.13 | 0.67 ± 0.04 | 0.98 ± 0.03 | -0.05 [-0.35, +0.18] |
| SAMD (teacher reset each round) | 0.79 ± 0.07 | 0.63 ± 0.04 | 0.71 ± 0.04 | 0.98 ± 0.01 | -0.07 [-0.26, +0.04] |
| lab method: reset + shared head + KD | 0.69 ± 0.06 | 0.60 ± 0.06 | 0.62 ± 0.17 | 1.00 ± 0.01 | -0.27 [-0.57, -0.06] * |
| reset + RKD | 0.67 ± 0.16 | 0.55 ± 0.07 | 0.71 ± 0.04 | 1.00 ± 0.00 | -0.29 [-0.49, -0.10] * |
| reset + PKT | 0.88 ± 0.06 | 0.71 ± 0.03 | 0.69 ± 0.12 | 0.97 ± 0.03 | +0.02 [-0.17, +0.20] |
| reset + SP | 0.73 ± 0.04 | 0.56 ± 0.05 | 0.69 ± 0.11 | 1.00 ± 0.00 | -0.25 [-0.61, +0.09] |
| control: reset schedule, no teaching term | 0.67 ± 0.10 | 0.58 ± 0.10 | 0.56 ± 0.11 | 1.00 ± 0.00 | -0.36 [-0.66, -0.12] * |
| control: reset + RKD, teacher sees photos only | 0.77 ± 0.22 | 0.61 ± 0.11 | 0.73 ± 0.10 | 0.99 ± 0.01 | -0.07 [-0.21, +0.09] |
| control: SAMD, teacher sees photos only | 0.71 ± 0.10 | 0.54 ± 0.03 | 0.77 ± 0.04 | 1.00 ± 0.00 | -0.30 [-0.55, -0.05] * |
| CAML distillation, teacher terms on scanned fruit only | 0.85 ± 0.13 | 0.68 ± 0.03 | 0.77 ± 0.07 | 0.97 ± 0.01 | +0.26 [-0.23, +0.79] |
| SAMD, teacher terms on scanned fruit only | 0.67 ± 0.04 | 0.57 ± 0.03 | 0.62 ± 0.06 | 1.00 ± 0.00 | -0.34 [-0.66, -0.09] * |
| lab method, teacher terms on scanned fruit only | 0.79 ± 0.24 | 0.65 ± 0.11 | 0.77 ± 0.10 | 0.98 ± 0.03 | +0.03 [-0.48, +0.35] |
| teacher: photo + microwave (needs the scanner) | 0.81 ± 0.33 | 0.68 ± 0.20 | 0.75 ± 0.06 | 0.97 ± 0.05 | +0.13 [-0.07, +0.32] |
| ours_rkd_b01_lr1e3 | 0.71 ± 0.04 | 0.60 ± 0.05 | 0.62 ± 0.06 | 1.00 ± 0.01 | -0.30 [-0.51, -0.11] * |
| ours_rkd_b01_lr3e3 | 0.71 ± 0.25 | 0.55 ± 0.15 | 0.79 ± 0.04 | 0.99 ± 0.01 | -0.15 [-0.28, -0.01] * |
| ours_rkd_b01_lr3e4 | 0.73 ± 0.10 | 0.61 ± 0.03 | 0.73 ± 0.10 | 1.00 ± 0.00 | -0.18 [-0.50, +0.13] |
| ours_rkd_b10_lr1e3 | 0.62 ± 0.11 | 0.51 ± 0.03 | 0.73 ± 0.04 | 1.00 ± 0.00 | -0.32 [-0.61, -0.07] * |
| ours_rkd_b10_lr3e3 | 0.79 ± 0.20 | 0.70 ± 0.19 | 0.60 ± 0.07 | 0.96 ± 0.02 | +0.15 [-0.04, +0.31] |
| ours_rkd_b10_lr3e4 | 0.79 ± 0.18 | 0.61 ± 0.04 | 0.73 ± 0.04 | 0.99 ± 0.02 | -0.16 [-0.35, +0.02] |
| ours_rkd_b1_lr1e3 | 0.58 ± 0.04 | 0.53 ± 0.02 | 0.67 ± 0.10 | 1.00 ± 0.00 | -0.34 [-0.60, -0.12] * |
| ours_rkd_b1_lr3e3 | 0.67 ± 0.04 | 0.55 ± 0.03 | 0.62 ± 0.06 | 0.99 ± 0.01 | -0.35 [-0.61, -0.11] * |
| ours_rkd_b1_lr3e4 | 0.73 ± 0.10 | 0.63 ± 0.03 | 0.69 ± 0.06 | 0.99 ± 0.01 | -0.13 [-0.54, +0.29] |
| samd_paired_b01_lr1e3 | 0.67 ± 0.10 | 0.58 ± 0.04 | 0.69 ± 0.11 | 1.00 ± 0.00 | -0.30 [-0.57, -0.06] * |
| samd_paired_b01_lr3e3 | 0.71 ± 0.04 | 0.61 ± 0.05 | 0.67 ± 0.10 | 0.98 ± 0.01 | -0.07 [-0.28, +0.09] |
| samd_paired_b01_lr3e4 | 0.83 ± 0.07 | 0.61 ± 0.05 | 0.77 ± 0.04 | 0.99 ± 0.02 | -0.10 [-0.42, +0.09] |
| samd_paired_b10_lr1e3 | 0.79 ± 0.13 | 0.63 ± 0.05 | 0.69 ± 0.12 | 0.99 ± 0.01 | -0.12 [-0.34, +0.11] |
| samd_paired_b10_lr3e3 | 0.73 ± 0.13 | 0.62 ± 0.08 | 0.60 ± 0.13 | 1.00 ± 0.00 | -0.35 [-0.62, -0.11] * |
| samd_paired_b10_lr3e4 | 0.88 ± 0.12 | 0.63 ± 0.03 | 0.77 ± 0.04 | 0.99 ± 0.02 | -0.04 [-0.39, +0.16] |
| samd_paired_b1_lr1e3 | 0.75 ± 0.06 | 0.64 ± 0.01 | 0.62 ± 0.17 | 1.00 ± 0.00 | -0.22 [-0.39, -0.07] * |
| samd_paired_b1_lr3e3 | 0.77 ± 0.04 | 0.63 ± 0.01 | 0.67 ± 0.07 | 0.99 ± 0.01 | -0.17 [-0.58, +0.21] |
| samd_paired_b1_lr3e4 | 0.81 ± 0.06 | 0.63 ± 0.04 | 0.75 ± 0.06 | 0.99 ± 0.02 | -0.08 [-0.41, +0.12] |

## latent_sde, tracked avocados only (1, 2, 3, 11) (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 0.77 ± 0.16 | 0.65 ± 0.09 | 0.67 ± 0.13 | 0.98 ± 0.02 | — |
| CAML distillation (alpha 0.9, t 4) | 0.92 ± 0.20 | 0.72 ± 0.17 | 0.77 ± 0.04 | 0.97 ± 0.03 | +0.27 [-0.04, +0.57] |
| CAML distillation (alpha 0.5, t 1) | 0.83 ± 0.14 | 0.64 ± 0.07 | 0.77 ± 0.10 | 1.00 ± 0.00 | -0.14 [-0.27, -0.02] * |
| feature matching (frozen teacher) | 0.75 ± 0.12 | 0.62 ± 0.06 | 0.73 ± 0.07 | 1.00 ± 0.00 | -0.19 [-0.37, +0.03] |
| CAML distillation + feature matching | 0.88 ± 0.38 | 0.71 ± 0.24 | 0.69 ± 0.06 | 0.97 ± 0.05 | +0.08 [-0.22, +0.43] |
| SAMD (teacher reset each round) | 0.73 ± 0.18 | 0.64 ± 0.08 | 0.60 ± 0.04 | 0.98 ± 0.02 | -0.19 [-0.43, +0.07] |
| teacher: photo + microwave (needs the scanner) | 0.96 ± 0.25 | 0.69 ± 0.18 | 0.77 ± 0.04 | 0.97 ± 0.05 | +0.19 [-0.17, +0.54] |

## latent_ode, all test predictions (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.36 ± 0.05 | 1.12 ± 0.04 | 0.73 ± 0.04 | 0.76 ± 0.01 | — |
| CAML distillation (alpha 0.9, t 4) | 1.41 ± 0.14 | 1.13 ± 0.08 | 0.66 ± 0.03 | 0.76 ± 0.01 | +0.03 [-0.03, +0.11] |
| SAMD (teacher reset each round) | 1.53 ± 0.11 | 1.26 ± 0.13 | 0.50 ± 0.22 | 0.76 ± 0.00 | -0.02 [-0.07, +0.05] |
| lab method: reset + shared head + KD | 1.53 ± 0.11 | 1.21 ± 0.03 | 0.62 ± 0.08 | 0.77 ± 0.01 | +0.04 [-0.03, +0.10] |
| reset + RKD | 1.49 ± 0.05 | 1.20 ± 0.02 | 0.57 ± 0.07 | 0.76 ± 0.00 | -0.06 [-0.10, -0.02] * |
| reset + SP | 1.45 ± 0.16 | 1.21 ± 0.04 | 0.58 ± 0.05 | 0.76 ± 0.01 | +0.01 [-0.04, +0.07] |
| control: reset schedule, no teaching term | 1.48 ± 0.21 | 1.22 ± 0.09 | 0.57 ± 0.05 | 0.76 ± 0.01 | +0.02 [-0.05, +0.12] |
| control: reset + RKD, teacher sees photos only | 1.49 ± 0.04 | 1.20 ± 0.04 | 0.56 ± 0.06 | 0.76 ± 0.01 | -0.05 [-0.10, -0.01] * |
| control: SAMD, teacher sees photos only | 1.54 ± 0.12 | 1.26 ± 0.13 | 0.50 ± 0.22 | 0.76 ± 0.00 | -0.02 [-0.08, +0.05] |
| SAMD, teacher terms on scanned fruit only | 1.50 ± 0.20 | 1.20 ± 0.16 | 0.62 ± 0.12 | 0.75 ± 0.01 | -0.01 [-0.08, +0.08] |
| reset + RKD, teacher terms on scanned fruit only | 1.51 ± 0.13 | 1.22 ± 0.11 | 0.58 ± 0.12 | 0.76 ± 0.01 | +0.05 [-0.05, +0.17] |
| teacher: photo + microwave (needs the scanner) | 1.42 ± 0.10 | 1.14 ± 0.06 | 0.74 ± 0.05 | 0.77 ± 0.01 | -0.03 [-0.07, +0.01] |

## latent_sde, all test predictions (teacher microwave input: 0+120+240)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.60 ± 0.09 | 1.24 ± 0.06 | 0.67 ± 0.08 | 0.75 ± 0.01 | — |
| CAML distillation (alpha 0.9, t 4) | 1.75 ± 0.36 | 1.35 ± 0.22 | 0.68 ± 0.08 | 0.73 ± 0.03 | +0.30 [+0.04, +0.53] * |
| CAML distillation (alpha 0.5, t 1) | 1.50 ± 0.12 | 1.23 ± 0.12 | 0.75 ± 0.08 | 0.75 ± 0.01 | +0.13 [-0.00, +0.26] |
| feature matching (frozen teacher) | 1.52 ± 0.11 | 1.24 ± 0.05 | 0.64 ± 0.05 | 0.77 ± 0.00 | +0.05 [-0.13, +0.31] |
| CAML distillation + feature matching | 1.57 ± 0.23 | 1.28 ± 0.18 | 0.62 ± 0.06 | 0.75 ± 0.02 | +0.09 [-0.12, +0.31] |
| relational KD (RKD, frozen teacher) | 1.60 ± 0.04 | 1.29 ± 0.02 | 0.64 ± 0.02 | 0.76 ± 0.01 | +0.12 [+0.02, +0.29] * |
| probabilistic KT (PKT, frozen teacher) | 1.54 ± 0.11 | 1.28 ± 0.09 | 0.59 ± 0.08 | 0.76 ± 0.02 | -0.01 [-0.17, +0.16] |
| similarity preserving (SP, frozen teacher) | 1.61 ± 0.23 | 1.27 ± 0.12 | 0.61 ± 0.05 | 0.75 ± 0.02 | +0.06 [-0.10, +0.27] |
| SAMD (teacher reset each round) | 1.57 ± 0.05 | 1.23 ± 0.02 | 0.61 ± 0.02 | 0.75 ± 0.01 | +0.05 [-0.14, +0.36] |
| lab method: reset + shared head + KD | 1.49 ± 0.08 | 1.20 ± 0.04 | 0.59 ± 0.11 | 0.75 ± 0.01 | -0.18 [-0.31, -0.06] * |
| reset + RKD | 1.44 ± 0.06 | 1.18 ± 0.02 | 0.60 ± 0.05 | 0.76 ± 0.01 | -0.17 [-0.29, -0.05] * |
| reset + PKT | 1.61 ± 0.07 | 1.30 ± 0.06 | 0.61 ± 0.07 | 0.75 ± 0.01 | +0.10 [-0.08, +0.38] |
| reset + SP | 1.52 ± 0.05 | 1.19 ± 0.05 | 0.61 ± 0.05 | 0.76 ± 0.01 | -0.14 [-0.31, +0.01] |
| control: reset schedule, no teaching term | 1.48 ± 0.11 | 1.21 ± 0.08 | 0.54 ± 0.10 | 0.75 ± 0.01 | -0.20 [-0.37, -0.03] * |
| control: reset + RKD, teacher sees photos only | 1.52 ± 0.13 | 1.21 ± 0.05 | 0.66 ± 0.03 | 0.75 ± 0.01 | +0.06 [-0.12, +0.38] |
| control: SAMD, teacher sees photos only | 1.42 ± 0.08 | 1.15 ± 0.03 | 0.66 ± 0.03 | 0.77 ± 0.01 | -0.16 [-0.31, -0.00] * |
| CAML distillation, teacher terms on scanned fruit only | 1.50 ± 0.17 | 1.22 ± 0.09 | 0.72 ± 0.11 | 0.75 ± 0.01 | +0.30 [+0.01, +0.60] * |
| SAMD, teacher terms on scanned fruit only | 1.47 ± 0.08 | 1.20 ± 0.07 | 0.56 ± 0.05 | 0.77 ± 0.01 | -0.21 [-0.37, -0.07] * |
| lab method, teacher terms on scanned fruit only | 1.51 ± 0.24 | 1.22 ± 0.15 | 0.71 ± 0.07 | 0.75 ± 0.01 | +0.08 [-0.13, +0.25] |
| teacher: photo + microwave (needs the scanner) | 1.61 ± 0.31 | 1.31 ± 0.20 | 0.66 ± 0.06 | 0.75 ± 0.02 | +0.17 [+0.04, +0.32] * |
| ours_rkd_b01_lr1e3 | 1.49 ± 0.02 | 1.22 ± 0.02 | 0.58 ± 0.05 | 0.76 ± 0.00 | -0.19 [-0.31, -0.07] * |
| ours_rkd_b01_lr3e3 | 1.51 ± 0.19 | 1.18 ± 0.10 | 0.69 ± 0.06 | 0.76 ± 0.00 | -0.05 [-0.16, +0.07] |
| ours_rkd_b01_lr3e4 | 1.49 ± 0.05 | 1.20 ± 0.03 | 0.68 ± 0.04 | 0.76 ± 0.01 | -0.12 [-0.27, +0.02] |
| ours_rkd_b10_lr1e3 | 1.38 ± 0.11 | 1.14 ± 0.03 | 0.64 ± 0.02 | 0.77 ± 0.00 | -0.18 [-0.33, -0.04] * |
| ours_rkd_b10_lr3e3 | 1.57 ± 0.13 | 1.29 ± 0.11 | 0.55 ± 0.02 | 0.74 ± 0.01 | +0.17 [+0.01, +0.32] * |
| ours_rkd_b10_lr3e4 | 1.55 ± 0.15 | 1.20 ± 0.06 | 0.69 ± 0.03 | 0.76 ± 0.01 | -0.10 [-0.21, +0.00] |
| ours_rkd_b1_lr1e3 | 1.36 ± 0.07 | 1.13 ± 0.04 | 0.64 ± 0.08 | 0.76 ± 0.01 | -0.20 [-0.34, -0.06] * |
| ours_rkd_b1_lr3e3 | 1.45 ± 0.02 | 1.19 ± 0.04 | 0.53 ± 0.08 | 0.76 ± 0.01 | -0.20 [-0.35, -0.05] * |
| ours_rkd_b1_lr3e4 | 1.49 ± 0.08 | 1.22 ± 0.04 | 0.70 ± 0.02 | 0.76 ± 0.00 | -0.13 [-0.30, +0.06] |
| samd_paired_b01_lr1e3 | 1.46 ± 0.04 | 1.21 ± 0.04 | 0.59 ± 0.08 | 0.77 ± 0.00 | -0.09 [-0.30, +0.22] |
| samd_paired_b01_lr3e3 | 1.49 ± 0.10 | 1.23 ± 0.05 | 0.58 ± 0.05 | 0.75 ± 0.01 | +0.12 [-0.09, +0.51] |
| samd_paired_b01_lr3e4 | 1.52 ± 0.11 | 1.22 ± 0.07 | 0.70 ± 0.02 | 0.76 ± 0.01 | -0.00 [-0.15, +0.14] |
| samd_paired_b10_lr1e3 | 1.55 ± 0.19 | 1.22 ± 0.09 | 0.62 ± 0.06 | 0.76 ± 0.01 | -0.01 [-0.18, +0.27] |
| samd_paired_b10_lr3e3 | 1.47 ± 0.11 | 1.22 ± 0.06 | 0.55 ± 0.09 | 0.75 ± 0.01 | -0.09 [-0.34, +0.34] |
| samd_paired_b10_lr3e4 | 1.59 ± 0.17 | 1.22 ± 0.10 | 0.71 ± 0.04 | 0.76 ± 0.01 | +0.09 [-0.10, +0.32] |
| samd_paired_b1_lr1e3 | 1.47 ± 0.08 | 1.21 ± 0.06 | 0.62 ± 0.08 | 0.77 ± 0.01 | -0.03 [-0.23, +0.36] |
| samd_paired_b1_lr3e3 | 1.52 ± 0.12 | 1.22 ± 0.06 | 0.64 ± 0.07 | 0.76 ± 0.01 | +0.01 [-0.26, +0.43] |
| samd_paired_b1_lr3e4 | 1.54 ± 0.13 | 1.22 ± 0.10 | 0.68 ± 0.08 | 0.76 ± 0.00 | +0.04 [-0.13, +0.25] |

## latent_sde, all test predictions (teacher microwave input: all)

| method | MAE (d) | CRPS | 80% cov. | C-index | log score vs photo only |
|---|---|---|---|---|---|
| photo only, no teacher (baseline) | 1.57 ± 0.11 | 1.24 ± 0.02 | 0.62 ± 0.08 | 0.75 ± 0.00 | — |
| CAML distillation (alpha 0.9, t 4) | 1.68 ± 0.24 | 1.33 ± 0.17 | 0.69 ± 0.05 | 0.75 ± 0.02 | +0.22 [+0.04, +0.39] * |
| CAML distillation (alpha 0.5, t 1) | 1.56 ± 0.03 | 1.24 ± 0.03 | 0.69 ± 0.05 | 0.75 ± 0.01 | -0.09 [-0.18, -0.01] * |
| feature matching (frozen teacher) | 1.46 ± 0.07 | 1.20 ± 0.04 | 0.68 ± 0.07 | 0.77 ± 0.00 | -0.05 [-0.20, +0.17] |
| CAML distillation + feature matching | 1.60 ± 0.32 | 1.29 ± 0.21 | 0.65 ± 0.02 | 0.74 ± 0.02 | +0.08 [-0.08, +0.25] |
| SAMD (teacher reset each round) | 1.48 ± 0.07 | 1.22 ± 0.02 | 0.58 ± 0.07 | 0.76 ± 0.01 | -0.01 [-0.24, +0.40] |
| teacher: photo + microwave (needs the scanner) | 1.70 ± 0.30 | 1.30 ± 0.20 | 0.68 ± 0.05 | 0.74 ± 0.02 | +0.21 [+0.03, +0.42] * |

