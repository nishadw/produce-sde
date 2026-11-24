#!/usr/bin/env bash
# One-photo phase 2 with filled lab training data: the fine-tune / teacher runs again with the training fruit extended
# by linear or logistic extrapolation (ripen.data.a2sl_synth, fit on training fruit only); fruit-only photos.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run/_snap.sh
for fill in linear logistic; do
  ids=()
  for m in latent_sde transformer latent_ode; do
    for s in 1 2 3; do
      ids+=("$(sub "opf-$fill-$m-s$s" "python scripts/run/one_photo_teacher.py --model $m --features fruit --seed $s --fill $fill" -c 4 --mem=16G -t 12:00:00)")
    done
  done
  echo "$fill: ${ids[*]}"
  sub "opf-$fill-summary" "python scripts/analysis/one_photo_teacher_compare.py outputs/one_photo_teacher_fill_$fill" -c 2 --mem=8G -t 1:00:00 --dependency=afterany:$(IFS=:; echo "${ids[*]}")
done
