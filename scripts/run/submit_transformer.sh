#!/usr/bin/env bash
# The transformer (ripen.models.sequence, configs/model/transformer.yaml) on the half-synthetic A2SL set, with the
# same settings as the latent ODE / SDE in submit_synth.sh: the grid (outputs/a2sl_synth_rs) and photo-only students
# with CAML-KD / SAMD, scanner or photo-only teacher (outputs/a2sl_synth_aml/<fill>, random start). 5 seeds x 4 folds.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run/_snap.sh
ids=()
for st in fixed random; do
  ids+=("$(sub "synth-transformer-time-$st" "python scripts/run/a2sl_synth_rs.py --model transformer --input time --synth linear --start $st")")
  for sy in linear logistic; do
    ids+=("$(sub "synth-transformer-photo-$sy-$st" "python scripts/run/a2sl_synth_rs.py --model transformer --input photo --synth $sy --start $st")")
  done
done
for sy in linear logistic; do
  A="python scripts/run/a2sl_aml.py data.scan_standardize=true data.synthetic=$sy train.random_start=true --model transformer --view all --eval-all-starts --out outputs/a2sl_synth_aml/$sy --seeds 1 2 3 4 5"
  for me in photo teacher kd samd; do ids+=("$(sub "saml-$sy-transformer-$me" "$A --method $me")"); done
  for me in kd samd; do ids+=("$(sub "saml-$sy-transformer-${me}_photoT" "$A --method $me --teacher-photo-only --tag _photoT")"); done
done
echo "${ids[*]}"
