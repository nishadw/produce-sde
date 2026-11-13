# Mango: views, time series and AML (`results/mango_views_recal`)

Cells: mean ± std over seeds; 6 folds by fruit; scored on the cut day without stage labels. Bootstrap: first minus reference, 95 % interval, `*` = excludes 0 (negative = better).

## References

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs known age + batch curve |
|---|---|---|---|---|---|
| known age + batch curve | 0.122 ± 0.000 | 0.071 ± 0.000 | -0.290 ± 0.000 | 0.329 ± 0.000 | — |
| photo + microwave clock | 0.125 ± 0.000 | 0.164 ± 0.000 | -0.335 ± 0.000 | 0.592 ± 0.000 | +0.003 [-0.025, +0.036] |
| microwave clock | 0.146 ± 0.000 | 0.174 ± 0.000 | -0.390 ± 0.000 | 0.584 ± 0.000 | +0.024 [-0.011, +0.064] |
| photo clock | 0.326 ± 0.000 | 0.124 ± 0.000 | -0.115 ± 0.000 | — | +0.204 [+0.144, +0.258] * |

## 1. Single views (latent SDE, one scan)

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs latent sde, photo, one scan |
|---|---|---|---|---|---|
| latent sde, photo, one scan | 0.259 ± 0.089 | 0.154 ± 0.000 | 0.028 ± 0.128 | — | — |
| latent sde, photo + microwave, image reconstruction, one scan | 0.189 ± 0.007 | 0.153 ± 0.006 | -0.148 ± 0.135 | 0.784 ± 0.022 | -0.070 [-0.096, -0.044] * |
| latent sde, photo + microwave, antennas 1 apart, one scan | 0.177 ± 0.009 | 0.166 ± 0.006 | -0.095 ± 0.034 | 0.786 ± 0.046 | -0.082 [-0.101, -0.063] * |
| latent sde, photo + microwave, antennas 2 apart, one scan | 0.176 ± 0.004 | 0.161 ± 0.010 | -0.053 ± 0.083 | 0.711 ± 0.010 | -0.083 [-0.110, -0.057] * |
| latent sde, photo + microwave, antennas 3 apart, one scan | 0.211 ± 0.016 | 0.184 ± 0.011 | -0.133 ± 0.074 | 0.817 ± 0.039 | -0.048 [-0.086, -0.000] * |
| latent sde, photo + microwave, antennas 4 apart, one scan | 0.172 ± 0.010 | 0.162 ± 0.016 | -0.021 ± 0.134 | 0.729 ± 0.085 | -0.087 [-0.108, -0.065] * |
| latent sde, photo + microwave, antennas 5 apart, one scan | 0.167 ± 0.009 | 0.171 ± 0.016 | -0.081 ± 0.193 | 0.687 ± 0.075 | -0.092 [-0.118, -0.066] * |
| latent sde, microwave, image reconstruction, one scan | 0.203 ± 0.000 | 0.165 ± 0.004 | -0.115 ± 0.097 | 0.804 ± 0.058 | -0.056 [-0.082, -0.027] * |
| latent sde, microwave, antennas 1 apart, one scan | 0.169 ± 0.008 | 0.160 ± 0.002 | -0.087 ± 0.039 | 0.756 ± 0.041 | -0.090 [-0.112, -0.067] * |
| latent sde, microwave, antennas 2 apart, one scan | 0.179 ± 0.007 | 0.166 ± 0.007 | -0.246 ± 0.074 | 0.727 ± 0.025 | -0.080 [-0.103, -0.055] * |
| latent sde, microwave, antennas 3 apart, one scan | 0.208 ± 0.009 | 0.168 ± 0.011 | -0.321 ± 0.127 | 0.813 ± 0.060 | -0.051 [-0.085, -0.006] * |
| latent sde, microwave, antennas 4 apart, one scan | 0.187 ± 0.010 | 0.151 ± 0.009 | -0.170 ± 0.075 | 0.868 ± 0.026 | -0.072 [-0.094, -0.045] * |
| latent sde, microwave, antennas 5 apart, one scan | 0.194 ± 0.008 | 0.175 ± 0.006 | -0.235 ± 0.028 | 0.779 ± 0.040 | -0.065 [-0.093, -0.033] * |
| latent sde, photo, photo side 1, one scan | 0.239 ± 0.047 | 0.161 ± 0.012 | -0.007 ± 0.143 | — | -0.020 [-0.041, +0.002] |
| latent sde, photo, photo side 2, one scan | 0.211 ± 0.016 | 0.154 ± 0.000 | -0.028 ± 0.351 | — | -0.048 [-0.069, -0.023] * |

## 2. All views in one model (latent SDE, one scan)

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs latent sde, photo, one scan |
|---|---|---|---|---|---|
| latent sde, photo + microwave, one scan | 0.186 ± 0.015 | 0.158 ± 0.014 | -0.225 ± 0.052 | 0.788 ± 0.037 | -0.072 [-0.099, -0.044] * |
| latent sde, photo + microwave, all six side by side, one scan | 0.177 ± 0.013 | 0.129 ± 0.006 | -0.064 ± 0.044 | 0.689 ± 0.020 | -0.082 [-0.116, -0.041] * |
| latent sde, photo + microwave, six separate models, combined (geometric mean), one scan | 0.172 ± 0.007 | 0.179 ± 0.010 | -0.106 ± 0.130 | 0.716 ± 0.025 | -0.087 [-0.110, -0.064] * |
| latent sde, photo + microwave, six separate models, combined (average), one scan | 0.168 ± 0.006 | 0.161 ± 0.008 | -0.124 ± 0.113 | 0.708 ± 0.024 | -0.091 [-0.113, -0.069] * |
| latent sde, photo + microwave, six separate models, combined (confidence-weighted), one scan | 0.170 ± 0.014 | 0.159 ± 0.022 | -0.124 ± 0.158 | 0.700 ± 0.026 | -0.089 [-0.117, -0.059] * |
| latent sde, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, one scan | 0.175 ± 0.018 | 0.167 ± 0.013 | -0.071 ± 0.093 | 0.779 ± 0.071 | -0.084 [-0.109, -0.059] * |
| latent sde, microwave, one scan | 0.186 ± 0.010 | 0.165 ± 0.009 | -0.263 ± 0.219 | 0.802 ± 0.020 | -0.073 [-0.095, -0.048] * |
| latent sde, microwave, all six side by side, one scan | 0.182 ± 0.015 | 0.149 ± 0.013 | -0.307 ± 0.122 | 0.718 ± 0.082 | -0.077 [-0.106, -0.042] * |
| latent sde, microwave, six separate models, combined (geometric mean), one scan | 0.184 ± 0.005 | 0.164 ± 0.015 | -0.307 ± 0.081 | 0.746 ± 0.010 | -0.075 [-0.096, -0.052] * |
| latent sde, microwave, six separate models, combined (average), one scan | 0.182 ± 0.002 | 0.161 ± 0.007 | -0.269 ± 0.077 | 0.727 ± 0.015 | -0.077 [-0.099, -0.053] * |
| latent sde, microwave, six separate models, combined (confidence-weighted), one scan | 0.193 ± 0.009 | 0.160 ± 0.001 | -0.234 ± 0.094 | 0.741 ± 0.006 | -0.066 [-0.101, -0.020] * |
| latent sde, microwave, 1, 2, 3, 4, 5 apart, pooled, one scan | 0.184 ± 0.012 | 0.166 ± 0.011 | -0.150 ± 0.113 | 0.799 ± 0.027 | -0.075 [-0.102, -0.046] * |

## 3. Time-series models with the scan history

Every instance sees all scans up to now; elapsed time is known, so the reference is the known-age batch curve.

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs known age + batch curve |
|---|---|---|---|---|---|
| gru d, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.135 ± 0.015 | 0.162 ± 0.007 | -0.113 ± 0.066 | 0.810 ± 0.045 | +0.013 [-0.016, +0.044] |
| gru d, microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.139 ± 0.009 | 0.158 ± 0.009 | -0.054 ± 0.037 | 0.830 ± 0.072 | +0.017 [-0.008, +0.048] |
| gru d, photo, scan history | 0.232 ± 0.005 | 0.181 ± 0.019 | -0.076 ± 0.032 | — | +0.110 [+0.081, +0.142] * |
| gru d, elapsed time only, scan history | 0.124 ± 0.002 | 0.096 ± 0.019 | -0.133 ± 0.122 | 0.391 ± 0.021 | +0.002 [-0.018, +0.022] |
| latent ode, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.104 ± 0.009 | 0.057 ± 0.007 | -0.099 ± 0.057 | 0.331 ± 0.011 | -0.018 [-0.035, -0.002] * |
| latent ode, microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.105 ± 0.004 | 0.057 ± 0.003 | -0.186 ± 0.121 | 0.327 ± 0.013 | -0.017 [-0.031, -0.003] * |
| latent ode, photo, scan history | 0.123 ± 0.017 | 0.070 ± 0.009 | -0.082 ± 0.113 | — | +0.001 [-0.015, +0.017] |
| latent ode, elapsed time only, scan history | 0.095 ± 0.008 | 0.048 ± 0.008 | -0.083 ± 0.062 | 0.288 ± 0.019 | -0.027 [-0.039, -0.016] * |
| latent sde, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.091 ± 0.012 | 0.056 ± 0.018 | 0.039 ± 0.223 | 0.334 ± 0.072 | -0.031 [-0.043, -0.019] * |
| latent sde, photo + microwave, antennas 5 apart, scan history | 0.088 ± 0.025 | 0.069 ± 0.016 | -0.012 ± 0.302 | 0.339 ± 0.064 | -0.034 [-0.042, -0.026] * |
| latent sde, microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.084 ± 0.037 | 0.046 ± 0.011 | 0.033 ± 0.092 | 0.282 ± 0.081 | -0.038 [-0.054, -0.024] * |
| latent sde, photo, scan history | 0.137 ± 0.030 | 0.087 ± 0.006 | -0.178 ± 0.185 | — | +0.015 [+0.001, +0.031] * |
| latent sde, elapsed time only, scan history | 0.097 ± 0.011 | 0.063 ± 0.005 | -0.151 ± 0.155 | 0.353 ± 0.038 | -0.025 [-0.036, -0.014] * |
| neural cde, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.145 ± 0.005 | 0.142 ± 0.010 | -0.238 ± 0.053 | 0.909 ± 0.137 | +0.023 [-0.001, +0.050] |
| neural cde, microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.155 ± 0.021 | 0.142 ± 0.008 | -0.177 ± 0.014 | 1.023 ± 0.244 | +0.033 [+0.007, +0.064] * |
| neural cde, photo, scan history | 0.190 ± 0.020 | 0.155 ± 0.009 | -0.130 ± 0.115 | — | +0.068 [+0.049, +0.092] * |
| neural cde, elapsed time only, scan history | 0.119 ± 0.001 | 0.066 ± 0.010 | -0.247 ± 0.137 | 0.329 ± 0.013 | -0.003 [-0.013, +0.007] |

## 3b. Same models, one scan

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs latent sde, photo, one scan |
|---|---|---|---|---|---|
| gru d, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, one scan | 0.153 ± 0.012 | 0.176 ± 0.012 | -0.174 ± 0.036 | 0.877 ± 0.060 | -0.106 [-0.136, -0.076] * |
| latent ode, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, one scan | 0.191 ± 0.027 | 0.173 ± 0.000 | -0.214 ± 0.109 | 1.068 ± 0.104 | -0.068 [-0.092, -0.041] * |
| neural cde, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, one scan | 0.151 ± 0.008 | 0.159 ± 0.022 | -0.151 ± 0.042 | 0.822 ± 0.063 | -0.108 [-0.136, -0.079] * |

## 4. AML: photo-only students (latent ode, photo, 1, 2, 3, 4, 5 apart, pooled, scan history)

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs latent ode, photo, scan history |
|---|---|---|---|---|---|
| latent ode, photo, scan history | 0.123 ± 0.017 | 0.070 ± 0.009 | -0.082 ± 0.113 | — | — |
| latent ode, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.104 ± 0.009 | 0.057 ± 0.007 | -0.099 ± 0.057 | 0.331 ± 0.011 | -0.019 [-0.039, +0.002] |
| latent ode, AML feat, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.123 ± 0.018 | 0.072 ± 0.006 | -0.101 ± 0.062 | — | -0.000 [-0.009, +0.009] |
| latent ode, AML kd, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.126 ± 0.025 | 0.072 ± 0.005 | -0.078 ± 0.052 | — | +0.003 [-0.006, +0.013] |
| latent ode, AML ours_pkt, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.108 ± 0.005 | 0.068 ± 0.005 | -0.097 ± 0.140 | — | -0.014 [-0.026, -0.003] * |
| latent ode, AML ours_rkd, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.117 ± 0.006 | 0.070 ± 0.005 | -0.126 ± 0.049 | — | -0.006 [-0.019, +0.008] |
| latent ode, AML ours_sp, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.109 ± 0.005 | 0.068 ± 0.005 | -0.093 ± 0.144 | — | -0.014 [-0.026, -0.003] * |
| latent ode, AML ours, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.106 ± 0.002 | 0.067 ± 0.005 | -0.097 ± 0.136 | — | -0.017 [-0.028, -0.006] * |
| latent ode, AML pkt, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.125 ± 0.024 | 0.072 ± 0.005 | -0.079 ± 0.036 | — | +0.002 [-0.006, +0.011] |
| latent ode, AML rkd, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.127 ± 0.019 | 0.072 ± 0.011 | -0.126 ± 0.175 | — | +0.004 [-0.007, +0.015] |
| latent ode, AML samd, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.107 ± 0.006 | 0.067 ± 0.006 | -0.013 ± 0.174 | — | -0.016 [-0.027, -0.005] * |
| latent ode, AML sp, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.125 ± 0.025 | 0.073 ± 0.005 | -0.076 ± 0.054 | — | +0.002 [-0.006, +0.012] |

