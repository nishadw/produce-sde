#!/usr/bin/env bash
# Storage-day model on the all-fruit A2SL study (docs/plans/2026-11-19_a2sl-all-fruit.md): photo and date only,
# random start, 12 leave-one-fruit-out folds, 5 seeds (deterministic, one job each) -> outputs/a2sl_all.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run/_snap.sh
G="python scripts/run/a2sl_synth_rs.py --synth real --split all --start random --out outputs/a2sl_all"
ids=()
for i in photo time; do ids+=("$(sub "all-storage_day-$i" "$G --model storage_day --input $i" -t 2:30:00)"); done
echo "${#ids[@]} jobs: ${ids[*]}"
sub all-summary "python scripts/analysis/a2sl_all_compare.py" -c 2 --mem=8G -t 2:00:00 --dependency=afterany:$(IFS=:; echo "${ids[*]}")
