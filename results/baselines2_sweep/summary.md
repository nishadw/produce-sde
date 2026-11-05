# Avocado multi-seed results: `results/baselines2_sweep`

Seeds: seed1. Cells are mean ± sample std over seeds; the split is fixed (same test items for every seed), so the spread is training variance only. Test split, every unspoiled prefix of every test item.

## Overall, every run

| run | seeds | MAE (d) | CRPS | 80% cov. | 80% width | C-index | ECE |
|---|---|---|---|---|---|---|---|
| sweep_cde_h32 | 1 | 1.617 | 1.170 | 0.847 | 4.877 | 0.889 | 0.042 |
| sweep_cde_h32_lr3e-4 | 1 | 1.622 | 1.158 | 0.909 | 5.960 | 0.893 | 0.027 |
| sweep_cde_h64 | 1 | 1.467 | 1.066 | 0.843 | 4.762 | 0.902 | 0.020 |
| sweep_gamma_process | 1 | 2.069 | 1.599 | 0.957 | 10.703 | 0.860 | 0.078 |
| sweep_gamma_process_lr1e-2 | 1 | 1.707 | 1.237 | 0.875 | 6.328 | 0.868 | 0.031 |
| sweep_gamma_process_lr1e-3 | 1 | 1.976 | 1.452 | 0.902 | 8.209 | 0.854 | 0.049 |
| sweep_gamma_process_lr3e-2 | 1 | 1.723 | 1.210 | 0.888 | 6.330 | 0.870 | 0.019 |
| sweep_landmark_cox | 1 | 4.144 | 2.771 | 0.993 | 17.914 | 0.751 | 0.128 |
| sweep_landmark_cox_lr1e-2 | 1 | 2.413 | 1.748 | 0.915 | 8.780 | 0.864 | 0.078 |
| sweep_landmark_cox_lr3e-2 | 1 | 2.251 | 1.648 | 0.922 | 8.391 | 0.866 | 0.065 |
| sweep_tf_d32_l1 | 1 | 0.922 | 0.705 | 0.959 | 4.454 | 0.949 | 0.019 |
| sweep_tf_d64_l2 | 1 | 0.964 | 0.693 | 0.934 | 3.654 | 0.947 | 0.013 |
| sweep_tf_d64_l2_lr3e-4 | 1 | 1.035 | 0.785 | 0.941 | 4.607 | 0.947 | 0.014 |
| sweep_tf_d64_l3 | 1 | 0.985 | 0.713 | 0.915 | 3.495 | 0.943 | 0.010 |
| sweep_wiener | 1 | 1.687 | 1.317 | 0.945 | 9.059 | 0.885 | 0.028 |
| sweep_wiener_lr1e-2 | 1 | 1.639 | 1.350 | 0.946 | 10.007 | 0.881 | 0.037 |
| sweep_wiener_lr1e-3 | 1 | 1.644 | 1.361 | 0.946 | 10.334 | 0.881 | 0.036 |
| sweep_wiener_lr3e-2 | 1 | 1.639 | 1.351 | 0.952 | 10.317 | 0.881 | 0.038 |

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
