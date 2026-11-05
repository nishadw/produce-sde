# Avocado multi-seed results: `results/avocado_round2`

Seeds: seed1, seed2, seed3, seed4, seed5. Cells are mean ± sample std over seeds; the split is fixed (same test items for every seed), so the spread is training variance only. Test split, every unspoiled prefix of every test item.

## Overall, every run

| run | seeds | MAE (d) | CRPS | 80% cov. | 80% width | C-index | ECE |
|---|---|---|---|---|---|---|---|
| fpt_filter_a2sl | 5 | 1.222 ± 0.079 | 1.019 ± 0.035 | 0.889 ± 0.000 | 5.200 ± 0.487 | 0.825 ± 0.017 | 0.155 ± 0.013 |
| gru_d_a2sl | 5 | 5.400 ± 0.217 | 3.514 ± 0.149 | 0.889 ± 0.000 | 17.800 ± 0.447 | 0.497 ± 0.039 | 0.319 ± 0.014 |
| latent_sde_a2sl_guided_r18 | 5 | 1.511 ± 0.310 | 1.287 ± 0.175 | 0.889 ± 0.000 | 4.978 ± 1.787 | 0.772 ± 0.014 | 0.136 ± 0.046 |
| latent_sde_avocado_every2_guided_r18 | 5 | 1.424 ± 0.075 | 1.121 ± 0.055 | 0.963 ± 0.010 | 5.987 ± 0.263 | 0.886 ± 0.002 | 0.025 ± 0.005 |
| latent_sde_avocado_every3_guided_r18 | 5 | 1.378 ± 0.070 | 0.973 ± 0.035 | 0.872 ± 0.031 | 4.635 ± 0.204 | 0.950 ± 0.003 | 0.010 ± 0.003 |
| latent_sde_avocado_guided_r18 | 5 | 1.302 ± 0.126 | 0.909 ± 0.061 | 0.936 ± 0.017 | 4.410 ± 0.284 | 0.933 ± 0.001 | 0.020 ± 0.007 |
| latent_sde_avocado_labels_only_guided_r18 | 5 | 1.319 ± 0.088 | 0.933 ± 0.039 | 0.936 ± 0.012 | 4.719 ± 0.151 | 0.930 ± 0.002 | 0.021 ± 0.004 |
| latent_sde_avocado_logo_T10_clock_fixed_guided_r18 | 5 | 8.891 ± 0.078 | 8.630 ± 0.064 | 0.002 ± 0.000 | 1.026 ± 0.061 | 0.808 ± 0.008 | 0.559 ± 0.006 |
| latent_sde_avocado_logo_T10_clock_group_guided_r18 | 5 | 8.253 ± 0.147 | 7.894 ± 0.150 | 0.007 ± 0.001 | 1.408 ± 0.044 | 0.834 ± 0.010 | 0.500 ± 0.014 |
| latent_sde_avocado_logo_T10_guided_r18 | 5 | 5.472 ± 0.200 | 4.743 ± 0.228 | 0.067 ± 0.015 | 2.984 ± 0.109 | 0.892 ± 0.004 | 0.271 ± 0.016 |
| latent_sde_avocado_logo_T20_clock_fixed_guided_r18 | 5 | 2.288 ± 0.269 | 1.570 ± 0.176 | 0.908 ± 0.065 | 8.377 ± 0.368 | 0.935 ± 0.003 | 0.158 ± 0.013 |
| latent_sde_avocado_logo_T20_clock_group_guided_r18 | 5 | 2.962 ± 0.223 | 2.226 ± 0.193 | 0.148 ± 0.061 | 3.883 ± 0.298 | 0.951 ± 0.003 | 0.240 ± 0.017 |
| latent_sde_avocado_logo_T20_guided_r18 | 5 | 1.078 ± 0.077 | 0.734 ± 0.055 | 0.824 ± 0.046 | 2.147 ± 0.081 | 0.954 ± 0.001 | 0.101 ± 0.009 |
| latent_sde_avocado_logo_Tamb_clock_fixed_guided_r18 | 5 | 3.290 ± 0.445 | 2.265 ± 0.327 | 0.657 ± 0.098 | 8.422 ± 0.876 | 0.931 ± 0.004 | 0.226 ± 0.027 |
| latent_sde_avocado_logo_Tamb_clock_group_guided_r18 | 5 | 4.221 ± 0.336 | 3.445 ± 0.326 | 0.002 ± 0.005 | 3.898 ± 0.097 | 0.942 ± 0.002 | 0.327 ± 0.019 |
| latent_sde_avocado_logo_Tamb_guided_r18 | 5 | 1.358 ± 0.084 | 0.937 ± 0.074 | 0.735 ± 0.047 | 2.829 ± 0.152 | 0.945 ± 0.001 | 0.130 ± 0.009 |
| latent_sde_avocado_mc_guided_r18 | 5 | 1.312 ± 0.058 | 0.929 ± 0.039 | 0.803 ± 0.031 | 2.851 ± 0.241 | 0.930 ± 0.002 | 0.033 ± 0.005 |

## Held-out storage temperature: MAE (d) and 80% coverage side by side

Coverage should be near 0.80; far below means the intervals are overconfident on the new condition.

| model | T10 MAE | T10 cov. | T20 MAE | T20 cov. | Tamb MAE | Tamb cov. |
|---|---|---|---|---|---|---|
| direct_reg | — | — | — | — | — | — |
| gru_d | — | — | — | — | — | — |
| fpt_filter | — | — | — | — | — | — |
| latent_sde | 5.472 ± 0.200 | 0.067 ± 0.015 | 1.078 ± 0.077 | 0.824 ± 0.046 | 1.358 ± 0.084 | 0.735 ± 0.047 |
| latent_sde+clock_group | 8.253 ± 0.147 | 0.007 ± 0.001 | 2.962 ± 0.223 | 0.148 ± 0.061 | 4.221 ± 0.336 | 0.002 ± 0.005 |
| latent_sde+clock_fixed | 8.891 ± 0.078 | 0.002 ± 0.000 | 2.288 ± 0.269 | 0.908 ± 0.065 | 3.290 ± 0.445 | 0.657 ± 0.098 |

## Clock ablation: MAE ratio (ablated / Arrhenius), mean over seeds

| held out | group / Arrhenius | fixed / Arrhenius |
|---|---|---|
| T10 | 1.51× | 1.62× |
| T20 | 2.75× | 2.12× |
| Tamb | 3.11× | 2.42× |
