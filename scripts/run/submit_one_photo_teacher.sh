#!/usr/bin/env bash
# One-photo phase 2 (docs/plans/2026-11-22_one-photo-app.md): the microwave teacher. Every job starts from the phase-1
# public model (outputs/one_photo/) and runs the 12 lab folds x 5 methods; then the summary.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run/_snap.sh
ids=()
for m in latent_sde transformer latent_ode; do
  for f in fruit colour; do
    for s in 1 2 3; do
      ids+=("$(sub "opt-$m-$f-s$s" "python scripts/run/one_photo_teacher.py --model $m --features $f --seed $s" -c 4 --mem=16G -t 12:00:00)")
    done
  done
done
echo "${#ids[@]} runs: ${ids[*]}"
sub opt-summary "python scripts/analysis/one_photo_teacher_compare.py" -c 2 --mem=8G -t 1:00:00 --dependency=afterany:$(IFS=:; echo "${ids[*]}")
