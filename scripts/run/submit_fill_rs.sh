#!/usr/bin/env bash
# Step 6 of docs/plans/2026-11-17_a2sl-imputation-random-start.md: smoke (tests + one short run), then the grid
# (latent ODE / SDE x {photo x fill none/linear/carry, date only} x start fixed/random, 5 seeds x 4 folds), then the
# summary. Each grid job waits for the smoke (afterok); the summary waits for every grid job (afterany).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ENV='source ./.ripen.env && source $RIPEN_ENV/bin/activate && export JAX_PLATFORMS=cpu'
sub() { local name="$1" cmd="$2"; shift 2
  sbatch --parsable -p class -A class -q default -c 2 --mem=16G -t 12:00:00 -J "ripen-$name" -o "outputs/logs/%x-%j.out" "$@" --wrap "$ENV && $cmd"; }
smoke=$(sub fillrs-smoke "python -m pytest -q tests/test_fill_start.py tests/test_a2sl.py tests/test_train_and_eval.py && python scripts/run/a2sl_fill_rs.py --model latent_ode --input photo --fill linear --start random --seeds 1 --folds 0 --out outputs/smoke_fill_rs train.epochs=2 && python scripts/analysis/a2sl_fill_rs_summary.py --root outputs/smoke_fill_rs")
echo "smoke $smoke"
deps=()
for m in latent_ode latent_sde; do
  for st in fixed random; do
    deps+=("$(sub "fillrs-$m-time-$st" "python scripts/run/a2sl_fill_rs.py --model $m --input time --start $st" --dependency=afterok:$smoke)")
    for f in none linear carry; do
      deps+=("$(sub "fillrs-$m-photo-$f-$st" "python scripts/run/a2sl_fill_rs.py --model $m --input photo --fill $f --start $st" --dependency=afterok:$smoke)")
    done
  done
done
echo "grid ${deps[*]}"
sub fillrs-summary "python scripts/analysis/a2sl_fill_rs_summary.py" -c 1 --mem=4G -t 0:30:00 --dependency=afterany:$(IFS=:; echo "${deps[*]}")
