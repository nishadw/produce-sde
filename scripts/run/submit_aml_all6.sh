#!/usr/bin/env bash
# CAML-style KD and SAMD with and without the scanner, teacher on all six views pooled (--view all), latent ODE / SDE,
# 5 seeds x 4 folds, fixed start, each test fruit also scored from every later start. -> outputs/a2sl_aml_all6
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ENV='source ./.ripen.env && source $RIPEN_ENV/bin/activate && export JAX_PLATFORMS=cpu'
OUT=outputs/a2sl_aml_all6
sub() { local name="$1" cmd="$2"; shift 2
  sbatch --parsable -p class -A class -q default -c 2 --mem=16G -t 12:00:00 -J "ripen-$name" -o "outputs/logs/%x-%j.out" "$@" --wrap "$ENV && $cmd"; }
deps=()
for m in latent_ode latent_sde; do
  A="python scripts/run/a2sl_aml.py data.scan_standardize=true --model $m --view all --eval-all-starts --out $OUT --seeds 1 2 3 4 5"
  for me in photo teacher kd samd; do deps+=("$(sub "aml6-$m-$me" "$A --method $me")"); done
  for me in kd samd; do deps+=("$(sub "aml6-$m-${me}_photoT" "$A --method $me --teacher-photo-only --tag _photoT")"); done
done
echo "jobs ${deps[*]}"
sub aml6-summary "python scripts/analysis/a2sl_fill_rs_summary.py --root $OUT" -c 1 --mem=4G -t 0:30:00 --dependency=afterany:$(IFS=:; echo "${deps[*]}")
