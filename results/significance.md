# Paired bootstrap: A minus B

Per-prediction CRPS and absolute error averaged over seeds, fruit resampled 5000 times. Negative = A is better. `*` = the 95 % interval excludes 0.

| comparison | fruit | predictions | Δ CRPS [95 % CI] | Δ MAE (d) [95 % CI] |
|---|---|---|---|---|
| in-dist_vs_transformer_avocado | 65 | 859 | +0.171 [+0.084, +0.281] * | +0.279 [+0.141, +0.419] * |
| in-dist_vs_neural_cde_avocado | 65 | 859 | -0.194 [-0.283, -0.102] * | -0.224 [-0.368, -0.078] * |
| in-dist_vs_wiener_avocado | 65 | 859 | -0.409 [-0.546, -0.251] * | -0.389 [-0.581, -0.175] * |
| in-dist_vs_gamma_process_avocado | 65 | 859 | -0.296 [-0.412, -0.151] * | -0.372 [-0.535, -0.185] * |
| in-dist_vs_landmark_cox_avocado | 65 | 859 | -0.751 [-0.930, -0.532] * | -1.005 [-1.282, -0.678] * |
| in-dist_vs_gru_gap_avocado | 65 | 859 | +0.068 [-0.016, +0.161] | +0.153 [+0.013, +0.294] * |
| in-dist_vs_ode_rnn_avocado | 65 | 859 | +0.078 [-0.015, +0.190] | +0.146 [+0.022, +0.281] * |
| in-dist_vs_gru_d_avocado | 65 | 859 | +0.012 [-0.078, +0.115] | +0.047 [-0.096, +0.200] |
| T10_vs_transformer | 155 | 3203 | -2.619 [-2.705, -2.528] * | -2.400 [-2.479, -2.316] * |
| T10_vs_wiener | 155 | 3203 | +2.542 [+2.370, +2.711] * | +2.439 [+2.291, +2.581] * |
| T10_vs_gamma_process | 155 | 3203 | -0.316 [-0.399, -0.232] * | -0.499 [-0.580, -0.419] * |
| T10_vs_gru_d | 155 | 3203 | -1.881 [-1.950, -1.807] * | -1.588 [-1.654, -1.517] * |
| T20_vs_transformer | 127 | 1122 | +0.194 [+0.149, +0.237] * | +0.362 [+0.303, +0.419] * |
| T20_vs_wiener | 127 | 1122 | -0.043 [-0.085, -0.001] * | -0.014 [-0.078, +0.048] |
| T20_vs_gamma_process | 127 | 1122 | -0.077 [-0.112, -0.042] * | -0.145 [-0.189, -0.102] * |
| T20_vs_gru_d | 127 | 1122 | +0.097 [+0.048, +0.145] * | +0.249 [+0.176, +0.320] * |
| Tamb_vs_transformer | 128 | 1049 | +0.228 [+0.198, +0.259] * | +0.472 [+0.416, +0.526] * |
| Tamb_vs_wiener | 128 | 1049 | +0.006 [-0.068, +0.085] | +0.042 [-0.065, +0.156] |
| Tamb_vs_gamma_process | 128 | 1049 | +0.181 [+0.109, +0.258] * | +0.279 [+0.183, +0.379] * |
| Tamb_vs_gru_d | 128 | 1049 | -0.443 [-0.486, -0.399] * | -0.432 [-0.495, -0.367] * |
| every2_vs_transformer | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every2_guided_r18 | |
| every2_vs_wiener | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every2_guided_r18 | |
| every2_vs_gru_d | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every2_guided_r18 | |
| every3_vs_transformer | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every3_guided_r18 | |
| every3_vs_wiener | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every3_guided_r18 | |
| every3_vs_gru_d | — | — | missing: no results/avocado_round2/seed*/latent_sde_avocado_every3_guided_r18 | |
