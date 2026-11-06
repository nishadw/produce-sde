# Progress

Current state as of 2026-11-05. The day-by-day log up to this point is in git history (`git log -p PROGRESS.md`).

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
  - Round 1, unguided, every model: `results/avocado_seeds/summary.md`, and `summary_recal.md` for the recalibrated version.
  - Round 2 + 2b, guided: `results/avocado_round2/summary.md`.
  - New baselines (Wiener, gamma, landmark Cox, neural CDE, transformer): `results/baselines2/summary.md`;
    sweep in `results/baselines2_sweep/`.
  - Paired bootstrap: `results/significance_recal.md`, `results/significance.md` (pairs in `results/significance_pairs.txt`).
  - Full report: `report/REPORT.md`.
- **For presenting:** `report/Produce_Spoilage_SDE_Results.docx` (rebuild: `report/brief/README.md`).
- **Tests:** 73 fast tests + smoke (all 14 models) pass. GitHub CI is green (fast job and slow recovery job).
- **Removed on 2026-11-05:** PBC2, C-MAPSS and DINOv2, plus the old `slurm/` scripts (replaced by `run_umiacs.sh`).
  - Their data and runs are archived in `/fs/class-projects/fall2026/cmsc498z/nwajge/scratch/archive_removed_2026-11-05`.

## In progress (2026-11-06): mango study
- Data arrived 2026-11-06 in `data/mango/` (raw, gitignored). Destructive design: 48 fruit, 6 cut per study day
  (0, 2, ..., 14); microwave scans every other day until the cut; photos and firmness on the cut day only.
  Spoiled = senescent (<= 1 N), seen only on days 12-14.
- Code: `scripts/ingest_mango.py`, `ripen.data.mango`, `ripen.train.samd` (SAMD distillation), `ripen.eval.mango`,
  `scripts/mango_cv.py`, `scripts/mango_summary.py`. 6 fast tests in `tests/test_mango.py`.
- Jobs: `bash run_umiacs.sh mango` -> 8218025-8218057 (8 models x {rgb, mw, both, samd} + references, CPU),
  summary 8218058 -> `results/mango/summary.md`. Then write `report/MANGO_REPORT.md`.

## Next steps, in order
1. **Mango study** (above): finish, report.
2. **Fix cold storage.** At T10 every learned model is 5.5–8.5 days off with coverage ≤ 0.16, while the Wiener process
   gets 3.0 days / 74 %. Options: a prior or hierarchy on the activation energy so clock uncertainty widens the
   intervals off-range; or a hybrid that leans on the per-fruit trend (Wiener-style) away from the training temperatures.
3. **Close the in-distribution gap to the transformer** (0.74 vs 0.85 CRPS): try the transformer as the latent SDE's
   encoder (`make_sequence_encoder("transformer", ...)` already exists).
4. **Ask the A2SL lab for more tracked fruit**, and labels that are not assigned by day (current ones leak the schedule).
5. **Full h-transform for the guided posterior** (backward filter on the PDE grid; closes the ~2-nat ELBO gap).
6. **Housekeeping decisions:**
   - The SDE-foundations material (`src/nsde/finance.py`, `experiments/local_vol_calibration.py`, `experiments/demo_video`,
     `assets/local_vol_*`) is from the CMSC 838B lecture and is unrelated to fruit. Keep it, or move it to its own repo.
   - Delete the scratch archive once nothing in it is needed.
