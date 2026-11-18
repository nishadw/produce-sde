#!/usr/bin/env bash
# Reruns after the KD-gradient fix (2026-11-17): the latent ODE's hard Monte Carlo forecast had zero gradient, so
# every latent ODE arm with forecast KD trained as if alpha were 0. The latent SDE (PDE forecast) was unaffected.
# Brief (outputs/a2sl_schedule): kd, ours, ours_rkd. Half-synthetic (outputs/a2sl_synth_aml): kd, kd_photoT.
# Then the two half-synthetic summaries. From a frozen snapshot of HEAD.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run/_snap.sh
m=latent_ode
A="python scripts/run/a2sl_aml.py data.scan_standardize=true --model $m --view 0+120+240 --out outputs/a2sl_schedule --seeds 1 2 3 4 5"
for me in kd ours ours_rkd; do echo "$(sub "kdfix-sched-$m-$me" "$A --method $me")"; done
for sy in linear logistic; do
  A="python scripts/run/a2sl_aml.py data.scan_standardize=true data.synthetic=$sy train.random_start=true --model $m --view all --eval-all-starts --out outputs/a2sl_synth_aml/$sy --seeds 1 2 3 4 5"
  ids=("$(sub "kdfix-saml-$sy-$m-kd" "$A --method kd")" "$(sub "kdfix-saml-$sy-$m-kd_photoT" "$A --method kd --teacher-photo-only --tag _photoT")")
  echo "${ids[*]}"
  sub "kdfix-saml-summary-$sy" "python scripts/analysis/a2sl_synth_rs_summary.py --root outputs/a2sl_synth_aml/$sy" -c 1 --mem=4G -t 0:30:00 --dependency=afterany:$(IFS=:; echo "${ids[*]}")
done
