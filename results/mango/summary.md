# Mango study: every model x input setting

Seeds: seed1, seed2, seed3; 6-fold cross-validation by fruit (each fruit is tested once per seed). Cells: mean ± std over seeds. Scored on each fruit's cut day, without stage labels (`ripen.eval.mango`).

## Main score: population Brier of the days-until-spoiled forecast (lower is better)

| model | photo only | photo only, SAMD-distilled | microwave only | photo + microwave |
|---|---|---|---|---|
| gru_gap | 0.200 ± 0.009 | 0.213 ± 0.019 | 0.147 ± 0.004 | 0.151 ± 0.006 |
| gru_d | 0.200 ± 0.009 | 0.213 ± 0.019 | 0.147 ± 0.004 | 0.151 ± 0.006 |
| ode_rnn | 0.200 ± 0.009 | 0.213 ± 0.019 | 0.147 ± 0.004 | 0.151 ± 0.006 |
| neural_cde | 0.197 ± 0.003 | 0.214 ± 0.010 | 0.154 ± 0.005 | 0.143 ± 0.014 |
| transformer | 0.199 ± 0.011 | 0.203 ± 0.011 | 0.152 ± 0.004 | 0.155 ± 0.011 |
| latent_sde | 0.262 ± 0.005 | 0.236 | 0.175 ± 0.003 | 0.177 ± 0.004 |
| latent_ode | 0.271 ± 0.012 | 0.220 ± 0.012 | 0.175 ± 0.017 | 0.184 ± 0.013 |
| fpt_filter | 0.329 ± 0.003 | 0.213 ± 0.004 | 0.212 ± 0.001 | 0.212 ± 0.001 |

References (closed form, no neural network):

| reference | population Brier | status Brier | Spearman vs firmness | within-day Spearman |
|---|---|---|---|---|
| known age + batch spoilage curve (oracle reference) | 0.077 ± 0.000 | 0.045 ± 0.000 | 0.927 ± 0.000 | -0.221 ± 0.000 |
| photo -> estimated age + batch curve | 0.182 ± 0.000 | 0.136 ± 0.000 | 0.844 ± 0.000 | -0.101 ± 0.000 |
| microwave -> estimated age + batch curve | 0.137 ± 0.000 | 0.171 ± 0.000 | 0.783 ± 0.000 | -0.314 ± 0.000 |
| photo + microwave -> estimated age + batch curve | 0.118 ± 0.000 | 0.161 ± 0.000 | 0.817 ± 0.000 | -0.288 ± 0.000 |

## Spoiled-now Brier on the cut day (lower is better)

| model | photo only | photo only, SAMD-distilled | microwave only | photo + microwave |
|---|---|---|---|---|
| gru_gap | 0.164 ± 0.016 | 0.186 ± 0.008 | 0.158 ± 0.010 | 0.138 ± 0.009 |
| gru_d | 0.164 ± 0.016 | 0.186 ± 0.008 | 0.158 ± 0.010 | 0.138 ± 0.009 |
| ode_rnn | 0.164 ± 0.016 | 0.186 ± 0.008 | 0.158 ± 0.010 | 0.138 ± 0.009 |
| neural_cde | 0.157 ± 0.008 | 0.170 ± 0.003 | 0.155 ± 0.007 | 0.136 ± 0.007 |
| transformer | 0.164 ± 0.032 | 0.174 ± 0.023 | 0.157 ± 0.007 | 0.154 ± 0.014 |
| latent_sde | 0.187 ± 0.001 | 0.161 | 0.183 ± 0.011 | 0.164 ± 0.004 |
| latent_ode | 0.188 ± 0.000 | 0.184 ± 0.001 | 0.186 ± 0.001 | 0.187 ± 0.001 |
| fpt_filter | 0.188 ± 0.000 | 0.187 ± 0.000 | 0.188 ± 0.000 | 0.188 ± 0.000 |

## Spearman: expected days left vs measured firmness (higher is better)

| model | photo only | photo only, SAMD-distilled | microwave only | photo + microwave |
|---|---|---|---|---|
| gru_gap | 0.254 ± 0.085 | 0.195 ± 0.056 | 0.614 ± 0.074 | 0.738 ± 0.024 |
| gru_d | 0.254 ± 0.085 | 0.195 ± 0.056 | 0.614 ± 0.074 | 0.738 ± 0.024 |
| ode_rnn | 0.254 ± 0.085 | 0.195 ± 0.056 | 0.614 ± 0.074 | 0.738 ± 0.024 |
| neural_cde | 0.274 ± 0.063 | 0.230 ± 0.095 | 0.672 ± 0.056 | 0.762 ± 0.073 |
| transformer | 0.224 ± 0.145 | 0.067 ± 0.085 | 0.634 ± 0.061 | 0.667 ± 0.110 |
| latent_sde | -0.021 ± 0.020 | 0.099 | 0.667 ± 0.047 | 0.638 ± 0.168 |
| latent_ode | 0.012 ± 0.053 | 0.070 ± 0.115 | 0.609 ± 0.091 | 0.386 ± 0.132 |
| fpt_filter | 0.181 ± 0.109 | 0.143 ± 0.102 | 0.050 ± 0.019 | 0.030 ± 0.070 |

## Within-day Spearman (fruit of the same age only; higher is better)

| model | photo only | photo only, SAMD-distilled | microwave only | photo + microwave |
|---|---|---|---|---|
| gru_gap | 0.055 ± 0.038 | -0.154 ± 0.187 | -0.213 ± 0.109 | -0.018 ± 0.125 |
| gru_d | 0.055 ± 0.038 | -0.154 ± 0.187 | -0.213 ± 0.109 | -0.018 ± 0.125 |
| ode_rnn | 0.055 ± 0.038 | -0.154 ± 0.187 | -0.213 ± 0.109 | -0.018 ± 0.125 |
| neural_cde | 0.046 ± 0.117 | -0.107 ± 0.222 | -0.153 ± 0.053 | -0.143 ± 0.194 |
| transformer | 0.077 ± 0.065 | -0.137 ± 0.182 | -0.165 ± 0.084 | -0.028 ± 0.148 |
| latent_sde | 0.026 ± 0.176 | 0.136 | -0.159 ± 0.127 | -0.221 ± 0.111 |
| latent_ode | -0.003 ± 0.124 | -0.088 ± 0.296 | -0.092 ± 0.107 | -0.027 ± 0.097 |
| fpt_filter | 0.030 ± 0.171 | -0.018 ± 0.076 | -0.374 ± 0.004 | -0.371 ± 0.065 |

## Mean predicted spoilage day (the batch spoiled between days 10 and 14)

| model | photo only | photo only, SAMD-distilled | microwave only | photo + microwave |
|---|---|---|---|---|
| gru_gap | 15.174 ± 0.211 | 15.750 ± 0.707 | 14.569 ± 0.464 | 14.285 ± 0.189 |
| gru_d | 15.174 ± 0.211 | 15.750 ± 0.707 | 14.569 ± 0.464 | 14.285 ± 0.189 |
| ode_rnn | 15.174 ± 0.211 | 15.750 ± 0.707 | 14.569 ± 0.464 | 14.285 ± 0.189 |
| neural_cde | 15.347 ± 0.295 | 15.757 ± 0.850 | 14.750 ± 0.199 | 14.208 ± 0.075 |
| transformer | 15.958 ± 0.566 | 16.097 ± 0.084 | 14.306 ± 0.407 | 14.403 ± 0.824 |
| latent_sde | 19.056 ± 1.284 | 12.938 | 14.438 ± 0.265 | 13.500 ± 0.324 |
| latent_ode | 19.639 ± 1.408 | 15.948 ± 0.398 | 15.146 ± 0.178 | 14.986 ± 0.690 |
| fpt_filter | 11.986 ± 0.012 | 15.090 ± 0.115 | 15.931 ± 0.024 | 15.986 ± 0.105 |

## Earlier scans (microwave settings): interval-censored log score, lower is better

| model | microwave only | photo + microwave |
|---|---|---|
| gru_gap | 0.759 ± 0.058 | 0.694 ± 0.052 |
| gru_d | 0.759 ± 0.058 | 0.694 ± 0.052 |
| ode_rnn | 0.759 ± 0.058 | 0.694 ± 0.052 |
| neural_cde | 0.842 ± 0.105 | 0.685 ± 0.095 |
| transformer | 0.771 ± 0.082 | 0.760 ± 0.062 |
| latent_sde | 0.734 ± 0.007 | 0.701 ± 0.018 |
| latent_ode | 0.777 ± 0.079 | 0.757 ± 0.107 |
| fpt_filter | 0.914 ± 0.001 | 0.909 ± 0.002 |

## Paired bootstrap on population Brier (first minus second; negative = first is better)

| model | SAMD − photo only | photo + microwave − photo only | SAMD − photo→age reference |
|---|---|---|---|
| gru_gap | +0.013 [-0.013, +0.040] | -0.049 [-0.078, -0.018] * | +0.030 [+0.002, +0.063] * |
| gru_d | +0.013 [-0.013, +0.040] | -0.049 [-0.078, -0.018] * | +0.030 [+0.002, +0.063] * |
| ode_rnn | +0.013 [-0.013, +0.040] | -0.049 [-0.078, -0.018] * | +0.030 [+0.002, +0.063] * |
| neural_cde | +0.017 [-0.014, +0.055] | -0.054 [-0.080, -0.026] * | +0.031 [-0.002, +0.069] |
| transformer | +0.004 [-0.009, +0.019] | -0.044 [-0.074, -0.011] * | +0.021 [-0.008, +0.054] |
| latent_sde | -0.026 [-0.098, +0.034] | -0.085 [-0.139, -0.036] * | +0.053 [+0.026, +0.084] * |
| latent_ode | -0.051 [-0.093, -0.011] * | -0.087 [-0.141, -0.035] * | +0.038 [-0.006, +0.094] |
| fpt_filter | -0.116 [-0.168, -0.052] * | -0.117 [-0.177, -0.044] * | +0.031 [-0.003, +0.076] |

## Other outputs on the cut day: firmness and ripeness class read out by ridge regression

| features | firmness MAE (N) | log-firmness MAE | class accuracy | class within one |
|---|---|---|---|---|
| photo | 6.608 ± 0.000 | 0.768 ± 0.000 | 0.375 ± 0.000 | 0.812 ± 0.000 |
| microwave | 5.718 ± 0.000 | 0.649 ± 0.000 | 0.438 ± 0.000 | 0.938 ± 0.000 |
| photo + microwave | 5.582 ± 0.000 | 0.620 ± 0.000 | 0.521 ± 0.000 | 0.938 ± 0.000 |
