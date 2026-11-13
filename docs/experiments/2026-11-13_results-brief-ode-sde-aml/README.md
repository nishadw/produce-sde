# 2026-11-13 · Results brief: latent ODE, latent SDE, AML

Plain-language version: [`results/2026-11-13_results-brief-ode-sde-aml.docx`](../../../results/2026-11-13_results-brief-ode-sde-aml.docx)
(built by `scripts/reports/brief_figs.py` then `brief_report.py`).

## What was run
- **Training-schedule ladder (avocado, 5 seeds × 4 leave-one-tracked-fruit-out folds, CPU).** Starting from a plain
  photo-only model, each ingredient of the reset-style (SAMD) schedule was added one at a time: early stopping on the
  all-visits validation loss (`photo_valall`), training in rounds (`rounds_only`), the take-turns copy step with a
  photo-only teacher (`reset_only --teacher-photo-only`), then the scanner teacher (`reset_only`, `samd`, `ours`,
  `ours_rkd`), plus CAML (`kd`, frozen teacher) and the teacher itself. Driver: `scripts/run/a2sl_aml.py ... --view
  0+120+240 --out outputs/a2sl_schedule --seeds 1 2 3 4 5`. Table: [`tables/avocado_training_schedule.md`](tables/avocado_training_schedule.md).
- **Mango correction fitted to the mango score** (`scripts/analysis/mango_recal_pop.py --compare`): isotonic, Platt
  and Platt + horizon maps, and a guarded choice by inner leave-one-fold-out. Table: [`tables/mango_corrections.md`](tables/mango_corrections.md).

## Findings
- The reset-style gain is the take-turns copy step: latent ODE log score 0.71 → 0.45 with no scanner anywhere, while
  MAE worsens 0.40 → 0.69 d (less over-confident, not more accurate). No teaching method beats that fair baseline.
- The scanner teacher itself is barely better than photo only (0.66 vs 0.71), so there is little to distil. CAML
  (no rounds) matches photo only.
- The earlier latent SDE gain (0.95 → 0.55) did not reproduce with 5 seeds (0.95 → 0.89).
- Mango: the score-fitted correction helps only weak models (one-scan photo 0.246 → 0.206) and hurts the good ones,
  so mango forecasts are reported uncorrected.
