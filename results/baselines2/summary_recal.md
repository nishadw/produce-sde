# Avocado multi-seed results: `results/baselines2` (post-hoc recalibrated on val)

Seeds: seed1, seed2, seed3, seed4, seed5. Cells are mean ± sample std over seeds; the split is fixed (same test items for every seed), so the spread is training variance only. Test split, every unspoiled prefix of every test item.

## Overall, every run

| run | seeds | MAE (d) | CRPS | 80% cov. | 80% width | C-index | ECE |
|---|---|---|---|---|---|---|---|
| neural_cde_a2sl | 5 | 2.333 ± 0.000 | 1.547 ± 0.018 | 0.667 ± 0.000 | 5.933 ± 0.149 | 0.500 ± 0.000 | 0.062 ± 0.006 |
| neural_cde_avocado | 5 | 1.514 ± 0.125 | 1.098 ± 0.079 | 0.875 ± 0.013 | 5.184 ± 0.358 | 0.898 ± 0.008 | 0.018 ± 0.005 |
| neural_cde_avocado_every2 | 5 | 2.933 ± 0.371 | 2.302 ± 0.053 | 0.862 ± 0.031 | 12.506 ± 0.746 | 0.782 ± 0.010 | 0.021 ± 0.004 |
| neural_cde_avocado_every3 | 5 | 2.591 ± 0.463 | 2.075 ± 0.272 | 0.564 ± 0.097 | 8.783 ± 3.411 | 0.892 ± 0.035 | 0.028 ± 0.016 |
| neural_cde_avocado_logo_T10 | 5 | 6.484 ± 0.107 | 6.156 ± 0.084 | 0.137 ± 0.010 | 1.532 ± 0.148 | 0.834 ± 0.003 | 0.286 ± 0.005 |
| neural_cde_avocado_logo_T20 | 5 | 1.694 ± 0.314 | 1.240 ± 0.234 | 0.907 ± 0.026 | 6.484 ± 1.117 | 0.858 ± 0.016 | 0.042 ± 0.017 |
| neural_cde_avocado_logo_Tamb | 5 | 1.741 ± 0.545 | 1.367 ± 0.438 | 0.808 ± 0.045 | 6.946 ± 1.324 | 0.884 ± 0.015 | 0.104 ± 0.008 |
| transformer_a2sl | 5 | 2.333 ± 0.000 | 1.589 ± 0.083 | 0.711 ± 0.099 | 5.822 ± 0.398 | 0.509 ± 0.021 | 0.078 ± 0.028 |
| transformer_avocado | 5 | 1.041 ± 0.079 | 0.739 ± 0.062 | 0.915 ± 0.013 | 3.686 ± 0.418 | 0.942 ± 0.003 | 0.023 ± 0.007 |
| transformer_avocado_every2 | 5 | 2.060 ± 0.382 | 1.475 ± 0.127 | 0.947 ± 0.011 | 8.770 ± 1.029 | 0.872 ± 0.015 | 0.037 ± 0.007 |
| transformer_avocado_every3 | 5 | 2.353 ± 0.984 | 1.821 ± 0.625 | 0.533 ± 0.177 | 5.557 ± 2.188 | 0.933 ± 0.018 | 0.018 ± 0.004 |
| transformer_avocado_logo_T10 | 5 | 7.769 ± 0.204 | 7.434 ± 0.292 | 0.138 ± 0.028 | 1.723 ± 0.555 | 0.728 ± 0.019 | 0.460 ± 0.033 |
| transformer_avocado_logo_T20 | 5 | 0.714 ± 0.032 | 0.577 ± 0.149 | 0.978 ± 0.011 | 4.813 ± 1.950 | 0.946 ± 0.005 | 0.051 ± 0.007 |
| transformer_avocado_logo_Tamb | 5 | 1.014 ± 0.432 | 0.877 ± 0.351 | 0.719 ± 0.048 | 5.071 ± 2.674 | 0.937 ± 0.010 | 0.093 ± 0.025 |

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
