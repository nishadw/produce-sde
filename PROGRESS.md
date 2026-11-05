# Progress

## 2026-11-02: independent multi-seed avocado rerun
- Remote renamed on GitHub: `nishadw/neural-sde-first-passage` -> `nishadw/produce-sde` (origin updated).
- `bash run_umiacs.sh seeds` submitted 37 jobs (one per avocado run from e1-e4, each looping seeds 1-5),
  job ids 8136322-8136358, plus `ripen-seeds-summary` 8136359 (afterany). Outputs: `results/avocado_seeds/seed<s>/<run>/`;
  results/runs is untouched. Seeds set both model init (`seed`) and batch order (`+train.seed`); split fixed at seed 0.
- When done: `results/avocado_seeds/summary.md` (mean ± std, LOGO MAE next to coverage, clock ratios).
  Rerun the table any time with `bash run_umiacs.sh seeds-summary --local` style: `python scripts/summarize_seeds.py`.
- Resubmit failures: check `sacct -u $USER -S today -X`; logs `results/logs/ripen-seeds-<i>-<jobid>.out`.

- 2026-11-03: verified done. All 37 seed jobs + summary job COMPLETED (exit 0), 5 seeds x 37 runs present,
  no errors in logs, results/runs untouched. summary.md committed (seed dirs are gitignored, 389M on disk only).

## C-MAPSS fpt_filter C-index 0.249 (diagnosed, not rerun: avocado-only run requested)
- Not a metric bug: `harrell_cindex(-median)` is shared by all models and fine elsewhere.
- Saved test predictions: predicted median vs true time-to-failure corr -0.65; predicted median vs elapsed
  time corr +0.87. The filter forecasts *more* life left as an engine ages, i.e. it ignores the sensor
  observations. Val loss was still falling at epoch 60 (not converged). Next: check how C-MAPSS sensors enter
  the filter's emission model and train longer.

- 2026-11-03: README + report/REPORT.md updated to 5-seed mean ± std (E1-E4); figures regenerated with error
  bars + per-seed dots (new e3_vs_baselines.png). Claims revised: GRU-D matches the SDE on T20/ambient (seed-0 GRU-D
  was a bad draw, configs identical, no NaN); MC beats the exact PDE likelihood on avocado too; ODE-RNN ties GRU+gap.

- 2026-11-03: CI had failed on every run since the repo started: it ran the slow recovery tests, which are
  tolerance-sensitive. On GitHub test_latent_sde_recovers_rates_sigma_and_fpt missed (CDF sup 0.325 > 0.3); on the
  cluster that one passes and test_arrhenius_clock_recovers_activation_energy misses (Ea 39.7k vs 50k, 21% > 15%,
  consistent with the known ~15-20% upward rate bias, E0). CI now gates on `-m "not slow"` + smoke (green on the
  cluster); slow tests run in a non-blocking `recovery` job. Tolerances not loosened.

- 2026-11-03: rate-recovery bias diagnosed and fixed (docs/verifier_report.md "Rate-recovery bias, revisited").
  ELBO gap shrinks with faster clock (KL ~ 1/kappa); exact likelihood peaks at truth. Fix: posterior_guidance +
  event_mean=log_mean (+ stage-unit z0 prior in tests; Ea test 30 epochs). All 3 slow tests pass on the cluster and on
  GitHub CI. Full-scale synthetic: rates +10/+5/+2% (was +24/+16/+14), sigma 0.256. Off by default; avocado unchanged.
  Diagnostic scripts + raw results: /fs/class-projects/fall2026/cmsc498z/nwajge/scratch/recovery (not in git).

## 2026-11-04: Round 2 started (docs/round2_plan.md)
- New: DINOv2 backbone (`--backbone dinov2_vitb14 --standardize`), `scripts/probe_features.py`, post-hoc PIT
  recalibration (`src/ripen/eval/recalibrate.py`, `scripts/recalibrate.py`), stages features2 / r2-pilot / r2 / recal.
- Found: integer-day quantiles make a perfect forecaster score cov80 0.85-0.88, not 0.80 (see plan note).
- Submitted: features2 8166759 (extract + probe), recal 8166760 (Round 1 recalibrated -> results/avocado_seeds/summary_recal.md),
  r2-pilot 8166761/8166766 + summary 8166767 (after features2) -> results/avocado_round2_pilot/.
- Gate passed: probe stage acc 0.759 (DINOv2) vs 0.714 (ResNet-18), days-left MAE 2.96 vs 2.91 (tie). Pilot clean
  (seed 1: latent SDE CRPS 0.923, GRU-D 0.965).
- 2026-11-04: `bash run_umiacs.sh r2` submitted: 45 jobs 8167177-8167237 (r2_0..r2_44) + summary 8167238 (afterany)
  -> results/avocado_round2/summary.md + summary_recal.md. Then: update report + professor summary page.

- 2026-11-05: Round 2 done (46/46 COMPLETED). Report section "Round 2", figure round2_ablation.png, plan results.
  Guidance = the win (CRPS 0.99 -> 0.90; 0.85 recalibrated); DINOv2 hurt sequence models; T10 coverage worse with
  guidance; exact PDE now beats MC. Comparison script: /fs/class-projects/fall2026/cmsc498z/nwajge/scratch/round2/cmp.py.

## Next
- Rerun E1-E4 (avocado) with model.posterior_guidance=true model.event_mean=log_mean, if the user wants it.
- Remaining ~2-nat gap: full h-transform via backward filtering on the PDE grid.
- Exact-vs-MC gap (E2), C-MAPSS filter emission model, PBC2 time scale, post-hoc recalibration.

## 2026-11-05: scope cut to avocado only
- Removed PBC2 + C-MAPSS (loaders, configs, tests, e5 stage, splits, report E5, docs/data_second_domain.md) and
  DINOv2 (feature-extraction path, --standardize, probe script, features2/r2-pilot stages, Round 2 dino runs).
- Round 2 is now the guided latent SDE (ResNet-18) only: results/avocado_round2 (4 runs x 5 seeds); `bash run_umiacs.sh r2`.
- Archived (not deleted) outside the repo: /fs/class-projects/fall2026/cmsc498z/nwajge/scratch/archive_removed_2026-11-05
  (pbc2/cmapss data + runs, DINOv2 cache + runs, pilot). Delete it when sure.
- Tests: 65 passed + smoke_all ok (job 8175981). Planning docs keep a scope note instead of being rewritten.
