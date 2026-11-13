# Avocado multi-seed results: `results/avocado/round2` (post-hoc recalibrated on val)

Seeds: seed1, seed2, seed3, seed4, seed5. Cells are mean ± sample std over seeds; the split is fixed (same test items for every seed), so the spread is training variance only. Test split, every unspoiled prefix of every test item.

## Overall, every run

| run | seeds | MAE (d) | CRPS | 80% cov. | 80% width | C-index | ECE |
|---|---|---|---|---|---|---|---|
| fpt_filter_a2sl | 5 | 1.178 ± 0.169 | 1.044 ± 0.017 | 0.667 ± 0.000 | 2.222 ± 0.208 | 0.834 ± 0.014 | 0.100 ± 0.008 |
| gru_d_a2sl | 5 | 2.333 ± 0.000 | 1.553 ± 0.007 | 0.667 ± 0.000 | 6.000 ± 0.000 | 0.500 ± 0.000 | 0.064 ± 0.002 |
| latent_sde_a2sl_guided_r18 | 5 | 1.578 ± 0.093 | 1.356 ± 0.096 | 0.511 ± 0.202 | 1.711 ± 0.420 | 0.778 ± 0.007 | 0.093 ± 0.009 |
| latent_sde_avocado_every2_guided_r18 | 5 | 1.516 ± 0.116 | 1.178 ± 0.038 | 0.962 ± 0.007 | 5.799 ± 0.282 | 0.886 ± 0.004 | 0.024 ± 0.002 |
| latent_sde_avocado_every3_guided_r18 | 5 | 1.390 ± 0.069 | 1.046 ± 0.055 | 0.767 ± 0.045 | 3.426 ± 0.198 | 0.950 ± 0.003 | 0.008 ± 0.002 |
| latent_sde_avocado_guided_r18 | 5 | 1.196 ± 0.025 | 0.850 ± 0.010 | 0.930 ± 0.005 | 4.291 ± 0.103 | 0.932 ± 0.001 | 0.018 ± 0.002 |
| latent_sde_avocado_labels_only_guided_r18 | 5 | 1.225 ± 0.064 | 0.900 ± 0.037 | 0.928 ± 0.005 | 4.487 ± 0.121 | 0.930 ± 0.001 | 0.020 ± 0.003 |
| latent_sde_avocado_logo_T10_clock_fixed_guided_r18 | 5 | 8.717 ± 0.018 | 8.650 ± 0.011 | 0.002 ± 0.000 | 0.685 ± 0.034 | 0.815 ± 0.001 | 0.555 ± 0.001 |
| latent_sde_avocado_logo_T10_clock_group_guided_r18 | 5 | 8.135 ± 0.113 | 7.897 ± 0.094 | 0.006 ± 0.001 | 0.983 ± 0.051 | 0.837 ± 0.011 | 0.491 ± 0.009 |
| latent_sde_avocado_logo_T10_guided_r18 | 5 | 5.290 ± 0.119 | 4.732 ± 0.139 | 0.060 ± 0.011 | 2.652 ± 0.116 | 0.893 ± 0.004 | 0.264 ± 0.012 |
| latent_sde_avocado_logo_T20_clock_fixed_guided_r18 | 5 | 4.501 ± 0.439 | 2.525 ± 0.162 | 0.910 ± 0.016 | 10.964 ± 0.372 | 0.935 ± 0.003 | 0.220 ± 0.017 |
| latent_sde_avocado_logo_T20_clock_group_guided_r18 | 5 | 3.282 ± 0.048 | 2.561 ± 0.112 | 0.045 ± 0.048 | 4.376 ± 0.339 | 0.951 ± 0.004 | 0.264 ± 0.003 |
| latent_sde_avocado_logo_T20_guided_r18 | 5 | 0.975 ± 0.035 | 0.599 ± 0.021 | 0.973 ± 0.012 | 2.436 ± 0.038 | 0.954 ± 0.001 | 0.082 ± 0.004 |
| latent_sde_avocado_logo_Tamb_clock_fixed_guided_r18 | 5 | 5.140 ± 0.548 | 3.148 ± 0.280 | 0.641 ± 0.050 | 10.776 ± 0.432 | 0.932 ± 0.002 | 0.281 ± 0.019 |
| latent_sde_avocado_logo_Tamb_clock_group_guided_r18 | 5 | 4.284 ± 0.099 | 3.634 ± 0.155 | 0.000 ± 0.000 | 4.358 ± 0.133 | 0.944 ± 0.001 | 0.338 ± 0.008 |
| latent_sde_avocado_logo_Tamb_guided_r18 | 5 | 1.467 ± 0.020 | 1.132 ± 0.029 | 0.545 ± 0.042 | 2.937 ± 0.065 | 0.945 ± 0.001 | 0.151 ± 0.005 |
| latent_sde_avocado_mc_guided_r18 | 5 | 1.191 ± 0.025 | 0.873 ± 0.007 | 0.920 ± 0.011 | 4.314 ± 0.198 | 0.932 ± 0.002 | 0.026 ± 0.012 |

## Held-out storage temperature: MAE (d) and 80% coverage side by side

Coverage should be near 0.80; far below means the intervals are overconfident on the new condition.

| model | T10 MAE | T10 cov. | T20 MAE | T20 cov. | Tamb MAE | Tamb cov. |
|---|---|---|---|---|---|---|
| direct_reg | — | — | — | — | — | — |
| gru_d | — | — | — | — | — | — |
| fpt_filter | — | — | — | — | — | — |
| latent_sde | 5.290 ± 0.119 | 0.060 ± 0.011 | 0.975 ± 0.035 | 0.973 ± 0.012 | 1.467 ± 0.020 | 0.545 ± 0.042 |
| latent_sde+clock_group | 8.135 ± 0.113 | 0.006 ± 0.001 | 3.282 ± 0.048 | 0.045 ± 0.048 | 4.284 ± 0.099 | 0.000 ± 0.000 |
| latent_sde+clock_fixed | 8.717 ± 0.018 | 0.002 ± 0.000 | 4.501 ± 0.439 | 0.910 ± 0.016 | 5.140 ± 0.548 | 0.641 ± 0.050 |

## Clock ablation: MAE ratio (ablated / Arrhenius), mean over seeds

| held out | group / Arrhenius | fixed / Arrhenius |
|---|---|---|
| T10 | 1.54× | 1.65× |
| T20 | 3.37× | 4.62× |
| Tamb | 2.92× | 3.50× |
