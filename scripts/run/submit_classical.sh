#!/usr/bin/env bash
# Weibull AFT and temporal image matching (ripen.models.classical) on the half-synthetic grid, the same settings as
# the latent ODE / SDE in submit_synth.sh, into outputs/a2sl_synth_rs; then the grid summary again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run/_snap.sh
ids=()
for m in weibull matching; do
  for st in fixed random; do
    ids+=("$(sub "synth-$m-time-$st" "python scripts/run/a2sl_synth_rs.py --model $m --input time --synth linear --start $st" -c 2 --mem=8G -t 3:00:00)")
    for sy in linear logistic; do
      ids+=("$(sub "synth-$m-photo-$sy-$st" "python scripts/run/a2sl_synth_rs.py --model $m --input photo --synth $sy --start $st" -c 2 --mem=8G -t 3:00:00)")
    done
  done
done
echo "${ids[*]}"
sub synth-summary-cls "python scripts/analysis/a2sl_synth_rs_summary.py --root outputs/a2sl_synth_rs" -c 1 --mem=4G -t 0:30:00 --dependency=afterany:$(IFS=:; echo "${ids[*]}")
