# Avocado multi-seed results: `results/baselines2_sweep` (post-hoc recalibrated on val)

Seeds: seed1. Cells are mean ± sample std over seeds; the split is fixed (same test items for every seed), so the spread is training variance only. Test split, every unspoiled prefix of every test item.

## Overall, every run

| run | seeds | MAE (d) | CRPS | 80% cov. | 80% width | C-index | ECE |
|---|---|---|---|---|---|---|---|
| sweep_cde_h32 | 1 | 1.516 | 1.091 | 0.895 | 5.283 | 0.893 | 0.023 |
| sweep_cde_h32_lr3e-4 | 1 | 1.526 | 1.096 | 0.888 | 5.441 | 0.894 | 0.015 |
| sweep_cde_h64 | 1 | 1.482 | 1.070 | 0.866 | 5.088 | 0.902 | 0.016 |
| sweep_gamma_process | 1 | 2.017 | 1.433 | 0.879 | 7.508 | 0.850 | 0.041 |
| sweep_landmark_cox | 1 | 3.806 | 2.459 | 0.909 | 13.796 | 0.754 | 0.024 |
| sweep_landmark_cox_lr1e-2 | 1 | 2.284 | 1.585 | 0.915 | 8.338 | 0.855 | 0.051 |
| sweep_landmark_cox_lr3e-2 | 1 | 2.157 | 1.516 | 0.895 | 7.764 | 0.863 | 0.027 |
| sweep_tf_d32_l1 | 1 | 0.988 | 0.699 | 0.920 | 3.649 | 0.945 | 0.028 |
| sweep_tf_d64_l2 | 1 | 0.962 | 0.695 | 0.938 | 3.737 | 0.944 | 0.017 |
| sweep_tf_d64_l2_lr3e-4 | 1 | 0.952 | 0.727 | 0.917 | 4.306 | 0.948 | 0.021 |
| sweep_tf_d64_l3 | 1 | 1.000 | 0.711 | 0.932 | 3.599 | 0.943 | 0.014 |
| sweep_wiener | 1 | 1.679 | 1.238 | 0.867 | 6.065 | 0.886 | 0.034 |

## Held-out storage temperature: MAE (d) and 80% coverage side by side

Coverage should be near 0.80; far below means the intervals are overconfident on the new condition.

| model | T10 MAE | T10 cov. | T20 MAE | T20 cov. | Tamb MAE | Tamb cov. |
|---|---|---|---|---|---|---|
| direct_reg | — | — | — | — | — | — |
| gru_d | — | — | — | — | — | — |
| fpt_filter | — | — | — | — | — | — |
| latent_sde | — | — | — | — | — | — |
| latent_sde+clock_group | — | — | — | — | — | — |
| latent_sde+clock_fixed | — | — | — | — | — | — |

## Clock ablation: MAE ratio (ablated / Arrhenius), mean over seeds

| held out | group / Arrhenius | fixed / Arrhenius |
|---|---|---|
| T10 | — | — |
| T20 | — | — |
| Tamb | — | — |
