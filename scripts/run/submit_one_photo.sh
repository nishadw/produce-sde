#!/usr/bin/env bash
# One-photo study (docs/plans/2026-11-22_one-photo-app.md): features job, then matching (deterministic, one run) and
# transformer / latent SDE / latent ODE x 3 seeds for each photo variant (full photo, fruit only, skin colour).
# Full-photo runs need no new features and start at once; the others wait for the features job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run/_snap.sh
feat=$(sub op-features "python scripts/data/one_photo_features.py" -c 4 --mem=32G -t 8:00:00)
echo "features: $feat"
ids=()
for f in full fruit colour; do
  dep=(); [[ $f != full ]] && dep=(--dependency=afterok:$feat)
  ids+=("$(sub "op-matching-$f" "python scripts/run/one_photo.py --model matching --features $f" -c 4 --mem=24G -t 3:00:00 "${dep[@]}")")
  for m in transformer latent_sde latent_ode; do
    for s in 1 2 3; do
      ids+=("$(sub "op-$m-$f-s$s" "python scripts/run/one_photo.py --model $m --features $f --seed $s" -c 4 -t 12:00:00 "${dep[@]}")")
    done
  done
done
echo "${#ids[@]} runs: ${ids[*]}"
sub op-summary "python scripts/analysis/one_photo_compare.py" -c 2 --mem=8G -t 1:00:00 --dependency=afterany:$(IFS=:; echo "${ids[*]}")
