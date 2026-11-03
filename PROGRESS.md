# Progress

## 2026-11-02: independent multi-seed avocado rerun
- Remote renamed on GitHub: `nishadw/neural-sde-first-passage` -> `nishadw/produce-sde` (origin updated).
- `bash run_umiacs.sh seeds` submitted 37 jobs (one per avocado run from e1-e4, each looping seeds 1-5),
  job ids 8136322-8136358, plus `ripen-seeds-summary` 8136359 (afterany). Outputs: `results/avocado_seeds/seed<s>/<run>/`;
  results/runs is untouched. Seeds set both model init (`seed`) and batch order (`+train.seed`); split fixed at seed 0.
- When done: `results/avocado_seeds/summary.md` (mean ± std, LOGO MAE next to coverage, clock ratios).
  Rerun the table any time with `bash run_umiacs.sh seeds-summary --local` style: `python scripts/summarize_seeds.py`.
- Resubmit failures: check `sacct -u $USER -S today -X`; logs `results/logs/ripen-seeds-<i>-<jobid>.out`.

## C-MAPSS fpt_filter C-index 0.249 (diagnosed, not rerun: avocado-only run requested)
- Not a metric bug: `harrell_cindex(-median)` is shared by all models and fine elsewhere.
- Saved test predictions: predicted median vs true time-to-failure corr -0.65; predicted median vs elapsed
  time corr +0.87. The filter forecasts *more* life left as an engine ages, i.e. it ignores the sensor
  observations. Val loss was still falling at epoch 60 (not converged). Next: check how C-MAPSS sensors enter
  the filter's emission model and train longer.

## Next
- When seeds finish: update README/report numbers with error bars; frame the SDE result as calibration +
  temperature transfer, not accuracy (ODE-RNN wins in-distribution in the seed-0 run).
