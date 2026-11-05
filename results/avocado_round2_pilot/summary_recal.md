# Avocado multi-seed results: `results/avocado_round2_pilot` (post-hoc recalibrated on val)

Seeds: seed1. Cells are mean ± sample std over seeds; the split is fixed (same test items for every seed), so the spread is training variance only. Test split, every unspoiled prefix of every test item.

## Overall, every run

| run | seeds | MAE (d) | CRPS | 80% cov. | 80% width | C-index | ECE |
|---|---|---|---|---|---|---|---|
| gru_d_avocado | 1 | 1.333 | 0.957 | 0.886 | 4.098 | 0.916 | 0.025 |
| latent_sde_avocado | 1 | 1.228 | 0.899 | 0.914 | 4.435 | 0.929 | 0.022 |

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
