#!/usr/bin/env bash
# Half-synthetic A2SL runs (docs/plans/2026-11-17_a2sl-imputation-random-start.md, revised design), from a frozen
# snapshot of HEAD: the main grid (outputs/a2sl_synth_rs) and CAML-KD / SAMD with and without the scanner
# (outputs/a2sl_synth_aml/<synth>, random start), 5 seeds x 4 folds, then one summary job per root.
# PART=grid or PART=aml submits only that part.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run/_snap.sh
grid=() aml_lin=() aml_log=()
for m in latent_ode latent_sde; do
  for st in fixed random; do
    [[ "${PART:-all}" == aml ]] && continue
    grid+=("$(sub "synth-$m-time-$st" "python scripts/run/a2sl_synth_rs.py --model $m --input time --synth linear --start $st")")
    for sy in linear logistic; do
      grid+=("$(sub "synth-$m-photo-$sy-$st" "python scripts/run/a2sl_synth_rs.py --model $m --input photo --synth $sy --start $st")")
    done
  done
  for sy in linear logistic; do
    [[ "${PART:-all}" == grid ]] && continue
    A="python scripts/run/a2sl_aml.py data.scan_standardize=true data.synthetic=$sy train.random_start=true --model $m --view all --eval-all-starts --out outputs/a2sl_synth_aml/$sy --seeds 1 2 3 4 5"
    ids=()
    for me in photo teacher kd samd; do ids+=("$(sub "saml-$sy-$m-$me" "$A --method $me")"); done
    for me in kd samd; do ids+=("$(sub "saml-$sy-$m-${me}_photoT" "$A --method $me --teacher-photo-only --tag _photoT")"); done
    if [[ $sy == linear ]]; then aml_lin+=("${ids[@]}"); else aml_log+=("${ids[@]}"); fi
  done
done
summ() { sub "$1" "python scripts/analysis/a2sl_synth_rs_summary.py --root $2" -c 1 --mem=4G -t 0:30:00 --dependency=afterany:$(IFS=:; shift 2; echo "$*"); }
echo "grid ${grid[*]}"; echo "aml ${aml_lin[*]} ${aml_log[*]}"
[[ ${#grid[@]} -gt 0 ]] && summ synth-summary outputs/a2sl_synth_rs "${grid[@]}"
if [[ ${#aml_lin[@]} -gt 0 ]]; then
  summ saml-summary-linear outputs/a2sl_synth_aml/linear "${aml_lin[@]}"
  summ saml-summary-logistic outputs/a2sl_synth_aml/logistic "${aml_log[@]}"
fi
