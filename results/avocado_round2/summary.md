# Avocado multi-seed results: `results/avocado_round2`

Seeds: seed1, seed2, seed3, seed4, seed5. Cells are mean ± sample std over seeds; the split is fixed (same test items for every seed), so the spread is training variance only. Test split, every unspoiled prefix of every test item.

## Overall, every run

| run | seeds | MAE (d) | CRPS | 80% cov. | 80% width | C-index | ECE |
|---|---|---|---|---|---|---|---|
| latent_sde_avocado_guided_r18 | 5 | 1.302 ± 0.126 | 0.909 ± 0.061 | 0.936 ± 0.017 | 4.410 ± 0.284 | 0.933 ± 0.001 | 0.020 ± 0.007 |
| latent_sde_avocado_logo_T10_guided_r18 | 5 | 5.472 ± 0.200 | 4.743 ± 0.228 | 0.067 ± 0.015 | 2.984 ± 0.109 | 0.892 ± 0.004 | 0.271 ± 0.016 |
| latent_sde_avocado_logo_T20_guided_r18 | 5 | 1.078 ± 0.077 | 0.734 ± 0.055 | 0.824 ± 0.046 | 2.147 ± 0.081 | 0.954 ± 0.001 | 0.101 ± 0.009 |
| latent_sde_avocado_logo_Tamb_guided_r18 | 5 | 1.358 ± 0.084 | 0.937 ± 0.074 | 0.735 ± 0.047 | 2.829 ± 0.152 | 0.945 ± 0.001 | 0.130 ± 0.009 |

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
