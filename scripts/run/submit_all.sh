#!/usr/bin/env bash
# All-fruit A2SL study (docs/plans/2026-11-19_a2sl-all-fruit.md): real photos only, random start, one fold per held-out
# fruit (avo01-12). Grid -> outputs/a2sl_all; CAML-KD / SAMD (scanner or photo-only teacher) -> outputs/a2sl_all_aml.
# Matching and Weibull are deterministic (all 5 seeds in one job); the neural models get one job per seed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run/_snap.sh
G="python scripts/run/a2sl_synth_rs.py --synth real --split all --start random --out outputs/a2sl_all"
A="python scripts/run/a2sl_aml.py data.scan_standardize=true train.random_start=true --split all --view all --eval-all-starts --out outputs/a2sl_all_aml"
ids=()
for m in matching weibull; do
  for i in photo time; do ids+=("$(sub "all-$m-$i" "$G --model $m --input $i" -t 6:00:00)"); done
done
for s in 1 2 3 4 5; do
  for m in latent_ode latent_sde transformer; do
    for i in photo time; do ids+=("$(sub "all-$m-$i-s$s" "$G --model $m --input $i --seeds $s" -t 24:00:00)"); done
    for me in teacher kd samd; do ids+=("$(sub "alla-$m-$me-s$s" "$A --model $m --method $me --seeds $s" -t 24:00:00)"); done
    for me in kd samd; do
      ids+=("$(sub "alla-$m-${me}_photoT-s$s" "$A --model $m --method $me --teacher-photo-only --tag _photoT --seeds $s" -t 24:00:00)")
    done
  done
done
echo "${#ids[@]} jobs: ${ids[*]}"
sub all-summary "python scripts/analysis/a2sl_all_compare.py" -c 2 --mem=8G -t 2:00:00 --dependency=afterany:$(IFS=:; echo "${ids[*]}")
