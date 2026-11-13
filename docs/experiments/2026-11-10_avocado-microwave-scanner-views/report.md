# A2SL avocados: which microwave view predicts best, and AML from microwave to photos

Requested by Prof. Ming (2026-11-07): a time-series model per microwave view, the best view, then AML / CAML
(photo + microwave teacher distilled to a photo-only student). Plain-language version: `summary.docx`
(`scripts/reports/a2sl_views_figs.py`, then `scripts/reports/a2sl_views_report.py`).

## Data and protocol

- `avocado_jun2023`: 15 fruit. Avocados 1, 2, 3, 11 are scanned at all 7 visits (days 1, 3, ..., 13); 4-10 are
  photographed daily and scanned only on their cut day; 12-15 have photos only. Every scan has 6 rotation views
  (0, 60, ..., 300°), each a 75x75 reconstruction; 35 scans, 25 after truncating at the first stage-5 visit.
- Labels are day-determined (`JMW2026_DAY_STAGE`); every tracked fruit spoils in (7, 9].
- 4 folds, leave one tracked fruit out (test = that fruit + ~3 cut fruit; val = 2 cut fruit); 3 seeds.
  `outputs/splits/a2sl_view_fold{k}.json`. Metrics pool the 4 folds' test forecasts per seed, then mean ± sd over
  seeds; significance = paired bootstrap over fruit of the censored log score.
- Microwave input: `scan_summary` (12 numbers per view: 4 moments + 8 radial bins); one view = those 12 + zeros,
  a pooled set = mean + sd over its views, `concat` = the six side by side (72).
- Recalibration (`--recal`): cross-fitted Kaplan-Meier PIT map fitted on the other 3 folds' test forecasts of the
  same run and seed, mixed 0.9 map + 0.1 raw forecast (`PITRecalibrator.fit(..., weight=0.9)`).

## Rounds

| round | change | where | outcome |
|---|---|---|---|
| 1 | as presented: labels are inputs | `outputs/a2sl_views/seed*`, `summary.md` | no sensor matters: the label history gives the answer; FPT forecasts identical to 1e-4 across views |
| 2 | labels hidden (targets only; guided posterior off) | `labels_hidden/` | latent SDE differs by view by about seed noise; FPT ignores the features |
| 2b | AML, teacher = photo + six pooled | `outputs/a2sl_aml/` | no student beats photo-only; 7/14 jobs died of CPU XLA mapped memory -> one process per seed x fold |
| 2c | recalibration: pure KM map -> 0.9 KM + 0.1 raw | `scratch/recal_diag.py` | mean log score better and the per-run blowups (up to 1.8 nats) gone, both studies |
| 3 | microwave numbers z-scored (`data.scan_standardize=true`) | `labels_hidden_std/`, `outputs/a2sl_aml_std/` | raw scans had mean ~2 / sd 0.5-5 vs photo ~0.3 and zero-filled missing scans; standardised, microwave-only beats the control at most views |
| 4 | pooled triples 0+120+240 / 60+180+300; AML teacher = photo + 0+120+240 | same roots | best combination; AML still no gain |

FPT (exact filter) was dropped after round 2: with labels hidden its photo/scan emission stays near-uniform, so every
view gives the same forecast (CRPS 1.29 microwave-only at every view, worse than its own time-only control).

## Results (latent SDE, round 3-4, tracked avocados 1/2/3/11)

`outputs/a2sl_views/labels_hidden_std/summary.md` / `summary_recal.md`; log score vs the time-only control, 95 %
bootstrap interval, * = excludes 0.

| input | MAE (d) | CRPS | log score vs control | recalibrated: vs control | recal. 80 % cov. |
|---|---|---|---|---|---|
| time only (control) | 1.10 | 0.73 | — | — | 0.77 |
| photo only | 0.85 | 0.62 | — | — | 0.73 |
| microwave 0° | 0.79 | 0.58 | -0.17 [-0.34, +0.01] | +0.03 | 0.79 |
| microwave 120° | 0.73 | 0.57 | -0.11 [-0.30, +0.02] | -0.01 | 0.75 |
| microwave 240° | 0.73 | 0.52 | -0.20 [-0.34, -0.03] * | -0.05 | 0.77 |
| microwave 60° / 180° / 300° | 0.98-1.17 | 0.70-0.73 | +0.01 / -0.05 / -0.05 | +0.26* / +0.13 / +0.05 | 0.71-0.77 |
| microwave 0+120+240 pooled | 0.77 | 0.58 | -0.12 [-0.28, +0.04] | -0.09 | 0.75 |
| microwave 60+180+300 pooled | 1.10 | 0.76 | -0.00 | +0.22 * | 0.75 |
| microwave six pooled | 0.85 | 0.61 | -0.14 [-0.30, +0.00] | -0.04 | 0.75 |
| photo + microwave 0+120+240 | 0.75 | 0.59 | -0.15 [-0.25, -0.05] * | +0.00 | 0.73 |
| photo + microwave six pooled (presented) | 0.79 | 0.63 | -0.07 [-0.18, +0.04] | +0.02 | 0.79 |
| photo + microwave six side by side | 1.08 | 0.75 | -0.01 | +0.01 | 0.73 |

**View groups.** Mean log score of the 0/120/240 views minus the 60/180/300 views, per fruit, paired bootstrap
(`scratch/evenodd.py`): negative in 16/16 cells and significant in 14/16 (rounds 2 and 3 x raw/recal x mw/photo_mw x
tracked/all), from -0.08 to -0.32. The one-scan snapshot probe (ridge from one scan to study day) does not show the
split (summary numbers: 0° 2.66 d vs 60° 2.43 d; pixels: 300° best), so the gap appears in tracking change over
visits, not in reading a single scan. A geometric cause (e.g. three antennas 120° apart, or interpolated
reconstructions at the odd angles) is unconfirmed; worth asking the lab.

**Snapshot** (`labels_hidden/summary.md`, last table): every view estimates the study day within ~2 days from one scan
(no scan: 3.45), with overlapping intervals.

## AML (teacher photo + microwave 0+120+240, `outputs/a2sl_aml_std/summary*.md`, tracked)

`ripen.train.aml.train_distill` (frozen teacher; `kd` = CAML loss `(1-a) L_task + a t^2 KL(q_T || q_S)` on the
days-until-spoiled pmf, a = 0.9, t = 4, plus a = 0.5, t = 1; `feat` = MSE to the teacher's representation; `kd_feat`);
`ripen.train.samd.train_samd` (SAMD; student and teacher now share a random visible prefix). Student = same visits,
photo features only.

| method | MAE (d) | CRPS | log score vs photo only | recalibrated |
|---|---|---|---|---|
| photo only (baseline) | 0.92 | 0.61 | — | — |
| CAML a 0.9, t 4 | 0.92 | 0.69 | +0.09 [-0.10, +0.26] | +0.32 |
| CAML a 0.5, t 1 | 1.06 | 0.69 | +0.08 [+0.01, +0.15] * | +0.12 |
| feature matching | 1.02 | 0.68 | +0.07 | -0.07 |
| CAML + feature matching | 0.79 | 0.64 | +0.03 | +0.10 |
| SAMD | 1.00 | 0.62 | -0.10 [-0.36, +0.15] | -0.07 |
| teacher (photo + microwave, needs scanner) | 0.96 | 0.68 | +0.07 [+0.01, +0.14] * | +0.13 |

Across the three AML rounds (teacher six pooled raw, six pooled z-scored, 0+120+240 z-scored) SAMD is the only
method on the better side every time (-0.08, -0.17, -0.10) but never significant; every other student is within
noise or worse. The teacher itself never beats photo-only, so there is no extra signal to distil.

## Why the gains are small

Every tracked fruit spoils in (7, 9] and visits are on a fixed schedule, so elapsed time alone pins the event to
about a day; sensors can only refine within that. With four tracked fruit, 3 seeds and fold-to-fold differences,
differences under ~0.1 nats are not resolvable. The photo-only baseline itself moves by 0.16 nats between two
identical configurations run with different seeds-to-key mappings (views study 0.37 vs AML driver 0.53,
recalibrated), which bounds what any comparison here can claim.

## Reproduce

```
RIPEN_SCAN_STD=1 RIPEN_VIEW_MODELS=latent_sde bash run_umiacs.sh a2sl-views
RIPEN_SCAN_STD=1 RIPEN_VIEW_MODELS=latent_sde RIPEN_AML_VIEW=0+120+240 bash run_umiacs.sh a2sl-aml
```
Rounds 1-2: `RIPEN_VIEW_LABELS=input` / default, without `RIPEN_SCAN_STD`.
