# Mango: views, time series and AML (`results/mango_views`)

Cells: mean ± std over seeds; 6 folds by fruit; scored on the cut day without stage labels. Bootstrap: first minus reference, 95 % interval, `*` = excludes 0 (negative = better).

## References

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs known age + batch curve |
|---|---|---|---|---|---|
| known age + batch curve | 0.077 ± 0.000 | 0.045 ± 0.000 | -0.221 ± 0.000 | 0.222 ± 0.000 | — |
| photo + microwave clock | 0.118 ± 0.000 | 0.161 ± 0.000 | -0.288 ± 0.000 | 0.573 ± 0.000 | +0.041 [+0.019, +0.066] * |
| microwave clock | 0.137 ± 0.000 | 0.171 ± 0.000 | -0.314 ± 0.000 | 0.579 ± 0.000 | +0.060 [+0.033, +0.092] * |
| photo clock | 0.182 ± 0.000 | 0.136 ± 0.000 | -0.101 ± 0.000 | — | +0.105 [+0.078, +0.133] * |

## 1. Single views (latent SDE, one scan)

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs latent sde, photo, one scan |
|---|---|---|---|---|---|
| latent sde, photo, one scan | 0.246 ± 0.005 | 0.187 ± 0.000 | 0.073 ± 0.159 | — | — |
| latent sde, photo + microwave, image reconstruction, one scan | 0.182 ± 0.005 | 0.154 ± 0.006 | -0.113 ± 0.151 | 0.755 ± 0.012 | -0.063 [-0.113, -0.019] * |
| latent sde, photo + microwave, antennas 1 apart, one scan | 0.170 ± 0.009 | 0.168 ± 0.009 | -0.058 ± 0.027 | 0.755 ± 0.035 | -0.075 [-0.126, -0.034] * |
| latent sde, photo + microwave, antennas 2 apart, one scan | 0.164 ± 0.004 | 0.161 ± 0.011 | -0.030 ± 0.093 | 0.690 ± 0.005 | -0.082 [-0.143, -0.034] * |
| latent sde, photo + microwave, antennas 3 apart, one scan | 0.202 ± 0.017 | 0.187 ± 0.009 | -0.084 ± 0.089 | 0.777 ± 0.015 | -0.043 [-0.106, +0.021] |
| latent sde, photo + microwave, antennas 4 apart, one scan | 0.162 ± 0.009 | 0.159 ± 0.019 | 0.008 ± 0.112 | 0.717 ± 0.041 | -0.083 [-0.136, -0.041] * |
| latent sde, photo + microwave, antennas 5 apart, one scan | 0.160 ± 0.010 | 0.165 ± 0.017 | -0.063 ± 0.194 | 0.691 ± 0.029 | -0.085 [-0.139, -0.042] * |
| latent sde, microwave, image reconstruction, one scan | 0.193 ± 0.002 | 0.167 ± 0.003 | -0.071 ± 0.090 | 0.778 ± 0.032 | -0.053 [-0.098, -0.008] * |
| latent sde, microwave, antennas 1 apart, one scan | 0.163 ± 0.005 | 0.162 ± 0.007 | -0.073 ± 0.036 | 0.747 ± 0.039 | -0.083 [-0.129, -0.044] * |
| latent sde, microwave, antennas 2 apart, one scan | 0.170 ± 0.005 | 0.167 ± 0.006 | -0.221 ± 0.086 | 0.707 ± 0.016 | -0.075 [-0.123, -0.036] * |
| latent sde, microwave, antennas 3 apart, one scan | 0.200 ± 0.005 | 0.171 ± 0.009 | -0.277 ± 0.139 | 0.802 ± 0.064 | -0.046 [-0.102, +0.017] |
| latent sde, microwave, antennas 4 apart, one scan | 0.176 ± 0.007 | 0.153 ± 0.012 | -0.108 ± 0.069 | 0.815 ± 0.037 | -0.069 [-0.114, -0.032] * |
| latent sde, microwave, antennas 5 apart, one scan | 0.186 ± 0.009 | 0.180 ± 0.003 | -0.188 ± 0.020 | 0.752 ± 0.014 | -0.060 [-0.109, -0.012] * |
| latent sde, photo, photo side 1, one scan | 0.244 ± 0.010 | 0.187 ± 0.001 | 0.038 ± 0.150 | — | -0.001 [-0.015, +0.011] |
| latent sde, photo, photo side 2, one scan | 0.245 ± 0.022 | 0.188 ± 0.000 | -0.008 ± 0.348 | — | -0.000 [-0.014, +0.013] |

## 2. All views in one model (latent SDE, one scan)

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs latent sde, photo, one scan |
|---|---|---|---|---|---|
| latent sde, photo + microwave, one scan | 0.177 ± 0.015 | 0.159 ± 0.018 | -0.200 ± 0.066 | 0.753 ± 0.027 | -0.068 [-0.129, -0.015] * |
| latent sde, photo + microwave, all six side by side, one scan | 0.169 ± 0.010 | 0.124 ± 0.010 | -0.058 ± 0.030 | 0.694 ± 0.001 | -0.076 [-0.143, -0.015] * |
| latent sde, photo + microwave, six separate models, combined (geometric mean), one scan | 0.165 ± 0.007 | 0.184 ± 0.009 | -0.072 ± 0.119 | 0.703 ± 0.019 | -0.081 [-0.134, -0.036] * |
| latent sde, photo + microwave, six separate models, combined (average), one scan | 0.161 ± 0.005 | 0.156 ± 0.010 | -0.067 ± 0.101 | 0.707 ± 0.012 | -0.085 [-0.139, -0.039] * |
| latent sde, photo + microwave, six separate models, combined (confidence-weighted), one scan | 0.162 ± 0.011 | 0.152 ± 0.022 | -0.084 ± 0.129 | 0.704 ± 0.012 | -0.084 [-0.144, -0.033] * |
| latent sde, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, one scan | 0.168 ± 0.016 | 0.170 ± 0.016 | -0.055 ± 0.101 | 0.755 ± 0.030 | -0.077 [-0.137, -0.029] * |
| latent sde, microwave, one scan | 0.175 ± 0.011 | 0.172 ± 0.008 | -0.228 ± 0.224 | 0.772 ± 0.009 | -0.070 [-0.114, -0.033] * |
| latent sde, microwave, all six side by side, one scan | 0.175 ± 0.011 | 0.148 ± 0.015 | -0.268 ± 0.117 | 0.706 ± 0.043 | -0.071 [-0.122, -0.023] * |
| latent sde, microwave, six separate models, combined (geometric mean), one scan | 0.175 ± 0.004 | 0.169 ± 0.018 | -0.263 ± 0.105 | 0.723 ± 0.008 | -0.071 [-0.114, -0.033] * |
| latent sde, microwave, six separate models, combined (average), one scan | 0.171 ± 0.002 | 0.159 ± 0.007 | -0.231 ± 0.089 | 0.728 ± 0.001 | -0.074 [-0.121, -0.034] * |
| latent sde, microwave, six separate models, combined (confidence-weighted), one scan | 0.179 ± 0.007 | 0.156 ± 0.003 | -0.200 ± 0.096 | 0.740 ± 0.006 | -0.067 [-0.124, -0.011] * |
| latent sde, microwave, 1, 2, 3, 4, 5 apart, pooled, one scan | 0.175 ± 0.009 | 0.170 ± 0.018 | -0.119 ± 0.098 | 0.776 ± 0.011 | -0.070 [-0.121, -0.026] * |

## 3. Time-series models with the scan history

Every instance sees all scans up to now; elapsed time is known, so the reference is the known-age batch curve.

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs known age + batch curve |
|---|---|---|---|---|---|
| gru d, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.136 ± 0.011 | 0.162 ± 0.004 | -0.059 ± 0.060 | 0.745 ± 0.028 | +0.058 [+0.032, +0.090] * |
| gru d, microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.136 ± 0.011 | 0.163 ± 0.005 | -0.005 ± 0.063 | 0.837 ± 0.103 | +0.059 [+0.037, +0.088] * |
| gru d, photo, scan history | 0.195 ± 0.008 | 0.161 ± 0.019 | 0.023 ± 0.047 | — | +0.118 [+0.102, +0.138] * |
| gru d, elapsed time only, scan history | 0.095 ± 0.003 | 0.083 ± 0.014 | -0.116 ± 0.142 | 0.297 ± 0.013 | +0.018 [+0.004, +0.034] * |
| latent ode, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.082 ± 0.006 | 0.045 ± 0.003 | -0.085 ± 0.073 | 0.293 ± 0.005 | +0.005 [-0.006, +0.016] |
| latent ode, microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.080 ± 0.002 | 0.042 ± 0.002 | -0.131 ± 0.138 | 0.281 ± 0.030 | +0.003 [-0.006, +0.013] |
| latent ode, photo, scan history | 0.100 ± 0.011 | 0.058 ± 0.011 | -0.037 ± 0.155 | — | +0.022 [+0.014, +0.032] * |
| latent ode, elapsed time only, scan history | 0.071 ± 0.005 | 0.040 ± 0.005 | -0.021 ± 0.017 | 0.253 ± 0.023 | -0.006 [-0.016, +0.003] |
| latent sde, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.083 ± 0.003 | 0.052 ± 0.013 | 0.013 ± 0.277 | 0.425 ± 0.025 | +0.006 [+0.001, +0.011] * |
| latent sde, photo + microwave, antennas 5 apart, scan history | 0.085 ± 0.005 | 0.057 ± 0.009 | 0.032 ± 0.309 | 0.427 ± 0.003 | +0.008 [+0.003, +0.013] * |
| latent sde, microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.088 ± 0.015 | 0.050 ± 0.006 | 0.063 ± 0.086 | 0.443 ± 0.014 | +0.011 [+0.005, +0.018] * |
| latent sde, photo, scan history | 0.113 ± 0.015 | 0.078 ± 0.006 | -0.161 ± 0.174 | — | +0.036 [+0.027, +0.045] * |
| latent sde, elapsed time only, scan history | 0.090 ± 0.007 | 0.057 ± 0.002 | -0.141 ± 0.111 | 0.436 ± 0.050 | +0.013 [+0.006, +0.019] * |
| neural cde, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.146 ± 0.001 | 0.144 ± 0.007 | -0.211 ± 0.065 | 0.855 ± 0.111 | +0.068 [+0.047, +0.095] * |
| neural cde, microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.155 ± 0.018 | 0.141 ± 0.005 | -0.150 ± 0.009 | 0.963 ± 0.202 | +0.077 [+0.054, +0.107] * |
| neural cde, photo, scan history | 0.177 ± 0.009 | 0.143 ± 0.007 | 0.028 ± 0.098 | — | +0.100 [+0.085, +0.119] * |
| neural cde, elapsed time only, scan history | 0.086 ± 0.002 | 0.057 ± 0.005 | -0.205 ± 0.172 | 0.258 ± 0.004 | +0.008 [+0.001, +0.016] * |

## 3b. Same models, one scan

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs latent sde, photo, one scan |
|---|---|---|---|---|---|
| gru d, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, one scan | 0.152 ± 0.010 | 0.175 ± 0.009 | -0.141 ± 0.055 | 0.828 ± 0.033 | -0.093 [-0.141, -0.055] * |
| latent ode, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, one scan | 0.188 ± 0.026 | 0.187 ± 0.001 | -0.180 ± 0.107 | 1.015 ± 0.096 | -0.057 [-0.097, -0.021] * |
| neural cde, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, one scan | 0.153 ± 0.008 | 0.158 ± 0.019 | -0.135 ± 0.052 | 0.770 ± 0.052 | -0.093 [-0.139, -0.057] * |

## 4. AML: photo-only students (latent ode, photo, 1, 2, 3, 4, 5 apart, pooled, scan history)

| run | population Brier | spoiled-now Brier | within-day Spearman | earlier-scan log score | pop. Brier vs latent ode, photo, scan history |
|---|---|---|---|---|---|
| latent ode, photo, scan history | 0.100 ± 0.011 | 0.058 ± 0.011 | -0.037 ± 0.155 | — | — |
| latent ode, photo + microwave, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.082 ± 0.006 | 0.045 ± 0.003 | -0.085 ± 0.073 | 0.293 ± 0.005 | -0.018 [-0.031, -0.003] * |
| latent ode, AML feat, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.099 ± 0.007 | 0.060 ± 0.009 | -0.035 ± 0.071 | — | -0.000 [-0.005, +0.005] |
| latent ode, AML kd, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.100 ± 0.012 | 0.059 ± 0.005 | -0.009 ± 0.108 | — | +0.000 [-0.005, +0.005] |
| latent ode, AML ours_pkt, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.104 ± 0.008 | 0.067 ± 0.008 | -0.068 ± 0.130 | — | +0.004 [-0.015, +0.025] |
| latent ode, AML ours_rkd, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.112 ± 0.014 | 0.069 ± 0.010 | -0.110 ± 0.008 | — | +0.012 [-0.009, +0.035] |
| latent ode, AML ours_sp, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.104 ± 0.008 | 0.067 ± 0.008 | -0.059 ± 0.133 | — | +0.004 [-0.015, +0.025] |
| latent ode, AML ours, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.104 ± 0.008 | 0.067 ± 0.008 | -0.057 ± 0.132 | — | +0.005 [-0.015, +0.025] |
| latent ode, AML pkt, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.099 ± 0.012 | 0.059 ± 0.006 | -0.003 ± 0.106 | — | -0.001 [-0.005, +0.004] |
| latent ode, AML rkd, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.100 ± 0.010 | 0.059 ± 0.004 | -0.067 ± 0.205 | — | +0.001 [-0.006, +0.008] |
| latent ode, AML samd, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.104 ± 0.009 | 0.067 ± 0.008 | 0.006 ± 0.130 | — | +0.004 [-0.015, +0.025] |
| latent ode, AML sp, 1, 2, 3, 4, 5 apart, pooled, scan history | 0.099 ± 0.012 | 0.059 ± 0.006 | -0.012 ± 0.098 | — | -0.001 [-0.005, +0.005] |

