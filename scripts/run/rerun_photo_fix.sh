#!/usr/bin/env bash
# Rerun every A2SL run that read photo features, after the 2026-11-17 photo-key fix (commit 1d78ff7):
# the views study (photo, photo + scanner; 3 seeds) and the AML training-schedule ladder (5 seeds).
# Old runs: outputs/_superseded_photo_key_bug_2026-11-17/. One 2-CPU job per run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
sub() {
  sbatch --parsable -p class -A class -q default -c 2 --mem=16G -t 8:00:00 -J "ripen-$1" -o "outputs/logs/%x-%j.out" \
    --wrap "source ./.ripen.env && source \$RIPEN_ENV/bin/activate && export JAX_PLATFORMS=cpu && $2"
}
V=outputs/a2sl_views/labels_hidden_std
for m in latent_ode latent_sde; do
  sub "fix-views-$m-photo" "python scripts/run/a2sl_views.py data.scan_standardize=true --model $m --input photo --labels hidden --out $V --seeds 1 2 3"
  for v in all concat 0+120+240 60+180+300 0 60 120 180 240 300; do
    sub "fix-views-$m-photo_mw-$v" "python scripts/run/a2sl_views.py data.scan_standardize=true --model $m --input photo_mw --view $v --labels hidden --out $V --seeds 1 2 3"
  done
  A="python scripts/run/a2sl_aml.py data.scan_standardize=true --model $m --view 0+120+240 --out outputs/a2sl_schedule --seeds 1 2 3 4 5"
  for me in photo photo_valall rounds_only reset_only samd ours ours_rkd kd teacher; do sub "fix-sched-$m-$me" "$A --method $me"; done
  sub "fix-sched-$m-reset_only_photoT" "$A --method reset_only --teacher-photo-only --tag _photoT"
  sub "fix-sched-$m-samd_photoT" "$A --method samd --teacher-photo-only --tag _photoT"
done
