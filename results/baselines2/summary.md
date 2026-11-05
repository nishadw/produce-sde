# Avocado multi-seed results: `results/baselines2`

Seeds: seed1, seed2, seed3, seed4, seed5. Cells are mean ± sample std over seeds; the split is fixed (same test items for every seed), so the spread is training variance only. Test split, every unspoiled prefix of every test item.

## Overall, every run

| run | seeds | MAE (d) | CRPS | 80% cov. | 80% width | C-index | ECE |
|---|---|---|---|---|---|---|---|
| neural_cde_a2sl | 5 | 5.178 ± 0.977 | 3.485 ± 0.298 | 0.889 ± 0.000 | 17.711 ± 0.442 | 0.512 ± 0.020 | 0.318 ± 0.031 |
| neural_cde_avocado | 5 | 1.526 ± 0.116 | 1.104 ± 0.080 | 0.877 ± 0.034 | 5.266 ± 0.808 | 0.898 ± 0.007 | 0.017 ± 0.003 |
| neural_cde_avocado_every2 | 5 | 2.794 ± 0.618 | 2.067 ± 0.322 | 0.840 ± 0.079 | 9.400 ± 1.446 | 0.785 ± 0.016 | 0.035 ± 0.018 |
| neural_cde_avocado_every3 | 5 | 2.614 ± 0.476 | 2.118 ± 0.288 | 0.602 ± 0.126 | 13.347 ± 2.821 | 0.893 ± 0.035 | 0.030 ± 0.023 |
| neural_cde_avocado_logo_T10 | 5 | 6.502 ± 0.133 | 6.168 ± 0.135 | 0.137 ± 0.007 | 1.566 ± 0.074 | 0.834 ± 0.003 | 0.290 ± 0.011 |
| neural_cde_avocado_logo_T20 | 5 | 1.529 ± 0.264 | 1.095 ± 0.184 | 0.861 ± 0.040 | 4.921 ± 0.457 | 0.864 ± 0.013 | 0.031 ± 0.009 |
| neural_cde_avocado_logo_Tamb | 5 | 1.531 ± 0.289 | 1.121 ± 0.207 | 0.818 ± 0.045 | 5.339 ± 1.412 | 0.888 ± 0.010 | 0.080 ± 0.015 |
| transformer_a2sl | 5 | 4.689 ± 1.095 | 3.147 ± 0.740 | 0.889 ± 0.000 | 15.422 ± 5.580 | 0.472 ± 0.042 | 0.283 ± 0.075 |
| transformer_avocado | 5 | 1.023 ± 0.100 | 0.738 ± 0.072 | 0.944 ± 0.020 | 4.250 ± 0.841 | 0.943 ± 0.004 | 0.020 ± 0.008 |
| transformer_avocado_every2 | 5 | 1.856 ± 0.382 | 1.377 ± 0.197 | 0.968 ± 0.013 | 9.347 ± 1.160 | 0.875 ± 0.015 | 0.038 ± 0.013 |
| transformer_avocado_every3 | 5 | 2.386 ± 0.999 | 1.893 ± 0.619 | 0.695 ± 0.164 | 14.579 ± 2.237 | 0.932 ± 0.018 | 0.023 ± 0.004 |
| transformer_avocado_logo_T10 | 5 | 7.872 ± 0.162 | 7.362 ± 0.238 | 0.156 ± 0.038 | 2.139 ± 0.646 | 0.740 ± 0.024 | 0.461 ± 0.026 |
| transformer_avocado_logo_T20 | 5 | 0.716 ± 0.025 | 0.540 ± 0.076 | 0.947 ± 0.022 | 3.325 ± 1.629 | 0.946 ± 0.004 | 0.051 ± 0.011 |
| transformer_avocado_logo_Tamb | 5 | 0.887 ± 0.289 | 0.709 ± 0.268 | 0.841 ± 0.039 | 3.660 ± 2.037 | 0.937 ± 0.008 | 0.069 ± 0.024 |

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
