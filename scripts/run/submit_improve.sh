#!/usr/bin/env bash
# All-fruit A2SL study, new photo features (docs/plans/2026-11-19_a2sl-all-fruit.md, follow-up 2026-11-22):
# fruit-only ResNet-18 (_fruit), skin colour (_colour) and the public-avocado ripeness reader (_public), built by
# scripts/data/a2sl_fruit_features.py. Matching (deterministic, one job) and latent ODE / SDE / transformer (one job per
# seed), photo input, random start, 12 folds -> outputs/a2sl_all. RIPEN_AFTER=<jobid> waits for the feature job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run/_snap.sh
G="python scripts/run/a2sl_synth_rs.py --synth real --split all --start random --input photo --out outputs/a2sl_all"
dep=(); [[ -n "${RIPEN_AFTER:-}" ]] && dep=(--dependency=afterok:$RIPEN_AFTER)
declare -A FEAT=([_fruit]=data/features/a2sl_fruit_resnet18.npz [_colour]=data/features/a2sl_colour.npz
                 [_public]=data/features/a2sl_public_stage.npz)
ids=()
for tag in _fruit _colour _public; do
  F="--features ${FEAT[$tag]} --tag $tag"
  ids+=("$(sub "imp-matching$tag" "$G --model matching $F" -t 2:30:00 "${dep[@]}")")
  for s in 1 2 3 4 5; do
    for m in latent_ode latent_sde transformer; do
      ids+=("$(sub "imp-$m$tag-s$s" "$G --model $m $F --seeds $s" -t 2:30:00 "${dep[@]}")")
    done
  done
done
echo "${#ids[@]} jobs: ${ids[*]}"
sub imp-summary "python scripts/analysis/a2sl_improve_compare.py" -c 2 --mem=8G -t 2:00:00 --dependency=afterany:$(IFS=:; echo "${ids[*]}")
