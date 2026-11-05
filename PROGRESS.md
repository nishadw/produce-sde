# Progress

Current state as of 2026-11-05. The day-by-day log up to this point is in git history (`git log -p PROGRESS.md`).

## Where things stand
- **Scope:** avocados only. The main data is the Mendeley photos (478 fruit). A small A2SL microwave-scan set
  (4 tracked fruit) is a demonstration.
- **Main model:** latent SDE with the guided posterior (`model.posterior_guidance=true model.event_mean=log_mean`)
  on ResNet-18 features, with post-hoc recalibration.
  - In-distribution CRPS 0.85 ± 0.01 (recalibrated), against GRU + gap 0.82 ± 0.04.
  - Held-out temperature MAE: 5.5 (T10), 1.1 (T20), 1.4 (ambient) days.
- **Results:**
  - Round 1, unguided, every model: `results/avocado_seeds/summary.md`, and `summary_recal.md` for the recalibrated version.
  - Round 2, guided: `results/avocado_round2/summary.md`.
  - Full report: `report/REPORT.md`.
- **For presenting:** `report/Produce_Spoilage_SDE_Results.docx` (rebuild: `report/brief/README.md`).
- **Tests:** 65 fast tests + smoke pass. GitHub CI is green (fast job and slow recovery job).
- **Removed on 2026-11-05:** PBC2, C-MAPSS and DINOv2, plus the old `slurm/` scripts (replaced by `run_umiacs.sh`).
  - Their data and runs are archived in `/fs/class-projects/fall2026/cmsc498z/nwajge/scratch/archive_removed_2026-11-05`.

## In progress (2026-11-05, started by the go-ahead)
- Step 1 submitted: `bash run_umiacs.sh r2b` -> jobs 8176371-8176383 (13 runs x 5 seeds) + summary 8176384
  -> results/avocado_round2/. Check: `sacct -u $USER -S today -X | grep r2b`.
- Next: code wiener / gamma / landmark Cox / Neural CDE / transformer baselines, pilot, then submit them (stage `nb`).

## Next steps, in order
1. **Finish Round 2 for the main model** (about 12 five-seed jobs, roughly half a day on 4 GPUs). Every number in the doc
   should come from one model version:
   - E3 clock ablation: guided latent SDE with `clock=group` and `clock=fixed` × 3 held-out groups.
   - E4 sparse photos: guided latent SDE, every 2nd and every 3rd day.
   - E2 likelihood: guided latent SDE with `event_likelihood=pde` and `=mc`.
   - A2SL: guided latent SDE, 5 seeds.

   Then regenerate the summaries, the figures, `report/brief/` and the docx.
2. **Ask the A2SL lab for more tracked fruit** now, because data requests take time. With 4 fruit the microwave column
   cannot support any claim.
3. **Fix cold storage.** At T10 every model is about 5 days off, with 80 % coverage of 0.05–0.20.
   - Put a prior on the activation energy, or make it hierarchical, so the clock's uncertainty widens the intervals
     off the training range.
   - Success means T10 coverage near 0.8 without hurting T20 or ambient.
4. **Classical first-passage baselines:** Wiener and gamma degradation processes. These are the textbook answer to
   "when does a degrading thing cross a threshold", so a reviewer will ask for them first.
5. **Modern sequence baselines:**
   - Neural CDE (fits our diffrax code).
   - A small transformer over the visit sequence.
6. **Full h-transform for the guided posterior.** Replace the one-label-ahead guidance with a backward filter on the
   PDE grid, which closes the remaining ~2-nat ELBO gap. This is research work; do it after 1–5.
7. **Housekeeping decisions:**
   - The SDE-foundations material (`src/nsde/finance.py`, `experiments/local_vol_calibration.py`, `experiments/demo_video`,
     `assets/local_vol_*`) is from the CMSC 838B lecture and is unrelated to fruit. Keep it, or move it to its own repo.
   - Delete the scratch archive once nothing in it is needed.
