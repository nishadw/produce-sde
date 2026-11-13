# Scanner views, time-series models and AML: A2SL avocados and mangoes

Plan: `docs/plans/views_aml_plan.md` (experiment IDs AV*/MG*/A*). Plain-language summary: `summary.docx`, built by
`scripts/reports/views_aml_figs.py` (ripen env) then `views_aml_report.py` (`envs/report/bin/python`).
Tables: `outputs/a2sl_views/labels_hidden_std/summary*.md`, `outputs/a2sl_aml_std/summary*.md`,
`outputs/mango_views/summary.md` (`_recal` for the recalibrated copy).

## Setup
- **Avocado (A2SL):** labels hidden as inputs, microwave numbers z-scored, 4 leave-one-tracked-avocado-out folds × 3
  seeds. Score: censored log score on the tracked avocados; bootstrap vs the time-only control (views) or the photo-only
  student (AML). AML teacher: photo + 0/120/240° pooled.
- **Mango:** new ingest `scripts/data/ingest_mango_views.py` → `data/mango/mango_scan_views.npz`: the 45 antenna pairs
  of the 10-antenna ring grouped by separation (sep1–sep5; 10/10/10/10/5 pairs, 80-d sweep summary each) plus the
  12-d image reconstruction. `load_mango(mw_view=..., photo_side=..., history=...)`. History mode: every instance sees
  all scans up to now at absolute study days. Score: cut-day population Brier (main), spoiled-now Brier, within-day
  Spearman, earlier-scan log score. 6 folds by fruit × 3 seeds.
- **AML methods** (`ripen.train.aml.METHODS`): kd (CAML / RepDistiller, γ .1, α .9, T 4), feat (FitNet MSE), kd_feat,
  rkd / pkt / sp (RepDistiller relational losses, rewritten for `[B, d]` features), samd (reset + shared head + MSE),
  ours (the lab's `train_ours_partial`: reset + shared head + KD), ours_rkd / ours_pkt / ours_sp, `--paired-only`
  (teacher terms only on scanned fruit; the lab's partial setting without its mismatched unpaired-batch KD term), and
  two controls: `reset_only` (reset schedule, no teaching term) and `--teacher-photo-only`.
- **Late fusion** (`scripts/analysis/late_fusion.py`, TMC-style): one model per view, forecasts combined by linear
  pool, log pool or exp(−entropy)-weighted pool.

## Results
**Avocado views (AV1/AV1b/AV2b).** Latent SDE: 0/120/240° are the useful views (as before). Latent ODE: all views within
run-to-run noise (photo + 0/120/240° 0.80, photo + all 0.79, photo only 0.82). Late fusion ≈ the pooled single model
(latent SDE photo_mw fuse-mean 0.96 vs all 0.96; ODE 0.82 vs 0.79). Neural CDE and GRU-D: best validation loss at epoch 0
on every fold (val rises from the start with 3 training fruit), so their test numbers are the untrained model
(MAE 3–4 d vs < 1 d for the latent models).

**Avocado models (AV3).** Latent ODE < latent SDE for every input (photo + 0/120/240°: 0.80 vs 0.88).

**Avocado AML (AV4/A11).** Frozen-teacher methods (kd, feat, rkd, pkt, sp, kd_paired): none beats photo-only (kd_paired
is significantly worse). Reset methods beat photo-only (latent SDE: ours_rkd 0.62, ours_sp 0.65, samd_paired 0.60 vs
0.95; latent ODE: samd 0.39, ours_rkd 0.43 vs 0.67). **Controls:** reset_only 0.55 (SDE) / 0.42 (ODE); SAMD with a
photo-only teacher 0.67 / 0.39. The gain is from the alternating schedule and shared head (likely extra,
differently-regularised training of the predictor), not from the microwave. The microwave teacher itself is never
better than photo-only. Grid (β ∈ {0.1, 1, 10} × lr ∈ {3e-4, 1e-3, 3e-3}, chosen on validation NLL): ours_rkd → β 1,
lr 3e-3 (0.44); samd_paired → β 1, lr 1e-3 (0.80; the identical config gave 0.60 in the first batch: repeats of one config
vary by ~0.2; jobs alternate GPU / CPU, which is not bit-identical, a likely source). β barely matters; lr does. The latent ODE samd and
samd_photoT forecasts agree to within 3/64 on every fold (forecasts are 64-sample empirical CDFs): the microwave
teacher barely moves the ODE student.

**Recalibration, avocado.** Cross-fitted PIT map (0.9 weight) improves most runs. After it, the SDE schedule gain
holds (photo 0.53 → reset_only 0.16), the ODE gain does not (photo 0.22, ours_rkd 0.17, reset_only 0.35).

**Mango views, one scan (MG1/MG2).** Every view beats photo (0.246). sep1/2/4/5 within 0.01 (best both + sep5 0.160);
sep3 worst (0.20); image reconstruction 0.18–0.19. Photo sides 1 and 2 identical. Pooled sep1–5 0.168, concat 0.169,
late fusion mean 0.161 / trust 0.162 / geo 0.165 (both). Nothing comes near the known-age reference (0.077).

**Mango time series (MG3).** With the scan history and known elapsed time: latent ODE time-only 0.071 (≈ age_clock
0.077, bootstrap n.s.), + microwave 0.080, + both 0.082, + photo 0.100. Latent ODE best for every input; latent SDE
0.083–0.113; neural CDE and GRU-D good with time only (0.086 / 0.095) but much worse once features are added
(0.136–0.195). One-scan neural CDE / GRU-D with pooled views (0.153 / 0.152) beat the one-scan latent SDE (0.168).

**Mango AML (MG4).** Latent ODE, history, teacher both + pooled sep1–5 (0.082). Every student within 0.012 of photo-only
(0.100); none significant.

**Recalibration, mango (MG5).** Hurts every run (age_clock 0.077 → 0.122). `scratch/mango_recal_diag.py` tried fit on
all / earlier-scan / cut-day instances × weight 0.9 / 0.5: all worse than raw except photo-only at weight 0.5. The PIT
map is fit on each fruit's own (mostly interval-censored) outcome, while the population Brier scores the CDF against
fold-mates' outcomes; the raw forecasts are already calibrated for that target. Mango numbers are reported raw.

**Follow-up: where the reset gain comes from (`outputs/a2sl_schedule`).** Photo only everywhere, 5 seeds × 4 folds,
all on CPU. Latent ODE log score / MAE: photo 0.71 / 0.40; + early stopping on the all-visits validation loss
(`photo_valall`) 0.61 / 0.46; + reset rounds without the teacher phase (`rounds_only`) 0.62 / 0.53; + teacher phase
with a photo-only teacher (`reset_only_photoT`, the fair baseline) 0.45 / 0.69; microwave teacher (`reset_only`)
0.44 / 0.68; SAMD 0.41 / 0.72; ours_rkd 0.48 / 0.72. Versus the fair baseline every AML method is within ±0.05
(only reset_only, −0.01, has an interval excluding 0). Latent SDE: photo 0.95, fair baseline 0.90, ours_rkd 0.86,
samd 0.87, none significant against the fair baseline; the first-round SDE gains (reset_only 0.55, ours_rkd 0.62) did
not reproduce. The extra predictor training in the teacher phase (fresh optimiser, shared head copied back) makes
forecasts wider: better log score, worse MAE, CRPS flat.

**Follow-up: mango correction on the population-Brier target (`outputs/mango_views_poprecal/maps.md`).** Map fitted on
the other folds' (F_i(d_j − d_i), s_j) pairs: isotonic, Platt, Platt + horizon, and a guarded choice by inner
leave-one-fold-out. Only weak runs improve (one-scan photo 0.246 → 0.206); good runs get worse (latent ODE time
0.071 → 0.083 isotonic); the guarded choice keeps them raw but is unstable with 48 fruit (latent ODE SAMD 0.104 →
0.125). Mango stays raw.

## Bugs found in the lab code (not ported)
- RepDistiller `train_ours_partial`: `loss_div_up` compares the student on an unpaired batch with the teacher on a
  different batch.
- TMC: hyperparameters selected on test accuracy.

## Ops
- CPU XLA "Failed to materialize symbols" after several seeds in one process → `mango_cv.py` runs one process per seed.
- One GPU job (ours_rkd β 1 lr 3e-3) hung for 8 h before its first log line on tron10; the resubmission ran normally.

**Scope (2026-11-13).** Neural CDE and GRU-D are dropped from the plain-language docs (summary.docx, brief.docx): with
3 training avocados per fold their best checkpoint was the untrained one, so they are not being pursued. Their runs
stay in `outputs/` and the tables above.
