# Progress

Current state as of 2026-11-05. Running log of what was done and where it stands, newest sections first in "Done".

## Where things stand
- **Scope:** avocados only. The main data is the Mendeley photos (478 fruit). A small A2SL microwave-scan set
  (4 tracked fruit) is a demonstration.
- **Main model:** latent SDE with the guided posterior (`model.posterior_guidance=true model.event_mean=log_mean`)
  on ResNet-18 features, with post-hoc recalibration. Round 2 is complete: every experiment uses this model.
  - In-distribution CRPS 0.85 ± 0.01 (recal.): tied with GRU + gap / ODE-RNN / GRU-D; transformer significantly better (0.74).
  - Held-out temperature MAE: 5.5 (T10), 1.1 (T20), 1.4 (ambient) days; Arrhenius clock 1.5–2.4× better than ablations.
  - Sparse photos (every 3rd day): CRPS 0.97 vs 1.5–2.7 for every other model; significant.
  - Cold storage: Wiener degradation process wins (3.0 days, 74 % coverage).
- **Results:**
  - Round 1, unguided, every model: `outputs/avocado/round1/summary.md`, and `summary_recal.md` for the recalibrated version.
  - Round 2 + 2b, guided: `outputs/avocado/round2/summary.md`.
  - New baselines (Wiener, gamma, landmark Cox, neural CDE, transformer): `outputs/avocado/baselines/summary.md`;
    sweep in `outputs/avocado/baselines_sweep/`.
  - Paired bootstrap: `outputs/avocado/significance/summary_recal.md`, `outputs/avocado/significance/summary.md` (pairs in `outputs/avocado/significance/pairs.txt`).
  - Full report: `docs/experiments/2026-10-28_avocado-latent-sde-vs-baselines/report.md`.
- **For presenting:** `results/2026-10-28_avocado-latent-sde-vs-baselines.docx` (rebuild: `scripts/reports/README.md`).
- **Tests:** 73 fast tests + smoke (all 14 models) pass. GitHub CI is green (fast job and slow recovery job).
- **Removed on 2026-11-05:** PBC2, C-MAPSS and DINOv2, plus the old `slurm/` scripts (replaced by `run_umiacs.sh`).
  - Their data and runs are archived in `/fs/class-projects/fall2026/cmsc498z/nwajge/scratch/archive_removed_2026-11-05`.

## Done (2026-11-12, later): training-schedule ladder + mango score-targeted correction + brief
- Presentation brief `results/2026-11-13_results-brief-ode-sde-aml.docx` (copy: `../spoilage results brief.docx`), built by
  `scripts/reports/brief_figs.py` then `brief_report.py`.
- Avocado ladder (`outputs/a2sl_schedule`, 5 seeds x 4 folds, all CPU; controls `photo_valall`, `rounds_only`,
  `reset_only --teacher-photo-only`): the reset gain is the take-turns copy step (latent ODE 0.71 -> 0.45), it improves
  the log score but worsens MAE (0.40 -> 0.69 d); no AML method beats that fair baseline; the latent SDE gain in the
  earlier round (0.95 -> 0.55) did not reproduce (0.95 -> 0.89).
- Mango: `scripts/analysis/mango_recal_pop.py` fits the map on population-Brier pairs (iso / Platt / guarded auto);
  only helps the weak models (`outputs/mango_views_poprecal/maps.md`); mango stays raw.

## Done (2026-11-12): views x time-series models x lab AML code, avocados + mangoes
- Plan `docs/experiments/2026-11-12_views-time-series-models-and-aml/experiment_sheet.md`; write-up `docs/experiments/2026-11-12_views-time-series-models-and-aml/report.md`; plain-language `summary.docx` (copy:
  `../views time series aml results.docx`), built by `scripts/reports/views_aml_figs.py` then `views_aml_report.py`.
- New: `ripen.train.aml` (RepDistiller losses, METHODS incl. lab `ours`, controls), `scripts/analysis/late_fusion.py`,
  mango per-separation views (`scripts/data/ingest_mango_views.py`), mango history mode, `scripts/analysis/mango_*`.
- Results: latent ODE best model on both fruits; avocado reset-style AML gains are reproduced by the schedule alone
  (no teacher / photo-only teacher); mango: sep3 worst, sep1/2/4/5 tied; with history, time only ≈ known age; mango
  AML nothing; mango recalibration hurts (reported raw).

## Done (2026-11-10): A2SL microwave views + AML (asked by Prof. Ming)
- Write-up `docs/experiments/2026-11-10_avocado-microwave-scanner-views/report.md`; plain-language `summary.docx` (copy: `../avocado views results.docx`),
  built by `scripts/reports/a2sl_views_figs.py` (ripen env) then `a2sl_views_summary.py` (envs/report/bin/python).
- Four rounds: labels as inputs (nothing matters) -> labels hidden -> z-scored microwave numbers + mixed recalibration
  -> pooled view triples. Views 0/120/240 beat 60/180/300 (14/16 significant); best input photo + 0/120/240.
  AML (CAML x2, feature matching, both, SAMD): no student beats photo-only; the teacher itself does not either.
- Results: `outputs/a2sl_views/labels_hidden{,_std}/summary*.md`, `outputs/a2sl_aml{,_std}/summary*.md`.
- Open: why 0/120/240 are better (ask the lab about scanner geometry); more tracked fruit / labels not set by day.

## Done (2026-11-06): mango study
- Data in `data/mango/` (raw, gitignored). Destructive design: 48 fruit, 6 cut per study day (0, 2, ..., 14);
  microwave scans every other day until the cut; photos and firmness on the cut day only. Spoiled = senescent.
- Code: `scripts/data/ingest_mango.py`, `ripen.data.mango`, `ripen.train.samd` (SAMD), `ripen.eval.mango`,
  `scripts/run/mango_cv.py`, `scripts/analysis/mango_summary.py`; tests in `tests/test_mango.py`. Run: `bash run_umiacs.sh mango`.
- Results: `outputs/mango/summary.md`; write-up `docs/experiments/2026-11-06_mango-photo-vs-microwave/report.md`. Microwave >> photo; SAMD does not beat the
  photo-only students (except weak structured ones); the photo->age->batch-curve reference beats every neural model.
- Ops note: on CPU, one process per seed (XLA code cache grows per fold; 4 latent jobs OOM'd before this).

## Next steps, in order
1. **Fix cold storage.** At T10 every learned model is 5.5–8.5 days off with coverage ≤ 0.16, while the Wiener process
   gets 3.0 days / 74 %. Options: a prior or hierarchy on the activation energy so clock uncertainty widens the
   intervals off-range; or a hybrid that leans on the per-fruit trend (Wiener-style) away from the training temperatures.
2. **Close the in-distribution gap to the transformer** (0.74 vs 0.85 CRPS): try the transformer as the latent SDE's
   encoder (`make_sequence_encoder("transformer", ...)` already exists).
3. **Ask the A2SL lab for more tracked fruit**, and labels that are not assigned by day (current ones leak the schedule).
4. **Full h-transform for the guided posterior** (backward filter on the PDE grid; closes the ~2-nat ELBO gap).
5. **Housekeeping decisions:**
   - The SDE-foundations material (`src/nsde/finance.py`, `foundations/experiments/local_vol_calibration.py`, `foundations/experiments/demo_video`,
     `foundations/assets/local_vol_*`) is from the CMSC 838B lecture and is unrelated to fruit. Keep it, or move it to its own repo.
   - Delete the scratch archive once nothing in it is needed.
