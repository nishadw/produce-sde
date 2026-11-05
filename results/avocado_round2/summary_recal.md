# Avocado multi-seed results: `results/avocado_round2` (post-hoc recalibrated on val)

Seeds: seed1, seed2, seed3, seed4, seed5. Cells are mean ± sample std over seeds; the split is fixed (same test items for every seed), so the spread is training variance only. Test split, every unspoiled prefix of every test item.

## Overall, every run

| run | seeds | MAE (d) | CRPS | 80% cov. | 80% width | C-index | ECE |
|---|---|---|---|---|---|---|---|
| latent_sde_avocado_guided_r18 | 5 | 1.196 ± 0.025 | 0.850 ± 0.010 | 0.930 ± 0.005 | 4.291 ± 0.103 | 0.932 ± 0.001 | 0.018 ± 0.002 |
| latent_sde_avocado_logo_T10_guided_r18 | 5 | 5.290 ± 0.119 | 4.732 ± 0.139 | 0.060 ± 0.011 | 2.652 ± 0.116 | 0.893 ± 0.004 | 0.264 ± 0.012 |
| latent_sde_avocado_logo_T20_guided_r18 | 5 | 0.975 ± 0.035 | 0.599 ± 0.021 | 0.973 ± 0.012 | 2.436 ± 0.038 | 0.954 ± 0.001 | 0.082 ± 0.004 |
| latent_sde_avocado_logo_Tamb_guided_r18 | 5 | 1.467 ± 0.020 | 1.132 ± 0.029 | 0.545 ± 0.042 | 2.937 ± 0.065 | 0.945 ± 0.001 | 0.151 ± 0.005 |

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
