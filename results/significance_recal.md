# Paired bootstrap: A minus B (both recalibrated)

Per-prediction CRPS and absolute error averaged over seeds, fruit resampled 5000 times. Negative = A is better. `*` = the 95 % interval excludes 0.

| comparison | fruit | predictions | Δ CRPS [95 % CI] | Δ MAE (d) [95 % CI] |
|---|---|---|---|---|
| in-dist_vs_transformer_avocado | 65 | 859 | +0.111 [+0.026, +0.220] * | +0.155 [+0.042, +0.276] * |
| in-dist_vs_neural_cde_avocado | 65 | 859 | -0.248 [-0.339, -0.150] * | -0.318 [-0.466, -0.157] * |
| in-dist_vs_wiener_avocado | 65 | 859 | -0.389 [-0.529, -0.226] * | -0.494 [-0.687, -0.274] * |
| in-dist_vs_gamma_process_avocado | 65 | 859 | -0.342 [-0.470, -0.186] * | -0.442 [-0.624, -0.235] * |
| in-dist_vs_landmark_cox_avocado | 65 | 859 | -0.672 [-0.842, -0.461] * | -0.983 [-1.253, -0.679] * |
| in-dist_vs_gru_gap_avocado | 65 | 859 | +0.028 [-0.052, +0.119] | +0.046 [-0.087, +0.190] |
| in-dist_vs_ode_rnn_avocado | 65 | 859 | +0.054 [-0.039, +0.166] | +0.060 [-0.068, +0.204] |
| in-dist_vs_gru_d_avocado | 65 | 859 | -0.025 [-0.117, +0.082] | -0.051 [-0.200, +0.112] |
| T10_vs_transformer | 155 | 3203 | -2.702 [-2.792, -2.608] * | -2.479 [-2.563, -2.391] * |
| T10_vs_wiener | 155 | 3203 | +2.565 [+2.387, +2.739] * | +2.282 [+2.129, +2.432] * |
| T10_vs_gamma_process | 155 | 3203 | -0.236 [-0.323, -0.147] * | -0.730 [-0.813, -0.646] * |
| T10_vs_gru_d | 155 | 3203 | -2.029 [-2.102, -1.952] * | -1.801 [-1.871, -1.724] * |
| T20_vs_transformer | 127 | 1122 | +0.021 [-0.026, +0.066] | +0.261 [+0.202, +0.316] * |
| T20_vs_wiener | 127 | 1122 | -0.178 [-0.216, -0.141] * | -0.242 [-0.307, -0.181] * |
| T20_vs_gamma_process | 127 | 1122 | -0.207 [-0.239, -0.176] * | -0.188 [-0.235, -0.141] * |
| T20_vs_gru_d | 127 | 1122 | -0.069 [-0.115, -0.024] * | +0.164 [+0.085, +0.239] * |
| Tamb_vs_transformer | 128 | 1049 | +0.255 [+0.224, +0.287] * | +0.453 [+0.393, +0.512] * |
| Tamb_vs_wiener | 128 | 1049 | +0.235 [+0.151, +0.323] * | +0.152 [+0.045, +0.263] * |
| Tamb_vs_gamma_process | 128 | 1049 | +0.236 [+0.162, +0.316] * | +0.231 [+0.142, +0.325] * |
| Tamb_vs_gru_d | 128 | 1049 | -0.328 [-0.376, -0.279] * | -0.460 [-0.530, -0.390] * |
| every2_vs_transformer | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every2_guided_r18 | |
| every2_vs_wiener | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every2_guided_r18 | |
| every2_vs_gru_d | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every2_guided_r18 | |
| every3_vs_transformer | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every3_guided_r18 | |
| every3_vs_wiener | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every3_guided_r18 | |
| every3_vs_gru_d | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every3_guided_r18 | |
