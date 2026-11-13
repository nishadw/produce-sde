# Round 2: the guided posterior on real data (avocado)

Started 2026-11-04. Round 1 = `outputs/avocado/round1/` (5 seeds, fixed split). Round 2 keeps the split, seeds 1–5,
epochs and configs, and changes two things:

| Change | Fixes | Applies to | How |
|---|---|---|---|
| **Guided posterior** (closed-form h-transform toward the next label) + `event_mean=log_mean` | E0: the unguided ELBO biases clock rates 14–24 % fast; fixed on synthetic (+2–10 %) but never used on real data | latent SDE, in-distribution + 3 LOGO folds | `model.posterior_guidance=true model.event_mean=log_mean` (`bash run_umiacs.sh r2`) |
| **Post-hoc PIT recalibration** (Kaplan–Meier on the PIT scale, fit on val, applied to test) | Over-wide intervals in-distribution (cov. 0.93–0.96) | every run, Round 1 and Round 2 (no retraining) | `scripts/analysis/recalibrate.py` → `metrics_test_recal.json` |

Scope change (2026-11-05): the project is avocado-only, ResNet-18 features. A DINOv2 photo-feature arm was tried
in this round and dropped (it made the sequence models worse); its code, feature cache and runs were removed.

## Hypotheses (written before the results)
- H2: Guidance sharpens the latent SDE's intervals (Round 1 width 5.2 d) without losing coverage.
- H3: Recalibration brings in-distribution coverage to the oracle level and lowers ECE, but does not fix T10
  (the validation PIT comes from the training temperatures).

## Note on the 80 % coverage target
`compute_metrics` takes quantiles on the integer-day grid and counts `q10 <= hi <= q90` inclusively, so a forecaster
that knows the true law scores **0.85–0.88**, not 0.80 (measured in `tests/test_recalibrate.py` style simulations with
true sd 0.7–3 days). Round 1's 0.93–0.96 is over-coverage, but by less than it looks; the latent ODE's 0.64 is worse
than it looks. Compare coverage against that oracle level, or use ECE.

## Results (5 November; details in docs/experiments/2026-10-28_avocado-latent-sde-vs-baselines/report.md, "Round 2")
- **H2 confirmed in-distribution, not under transfer.**
  - Guidance takes CRPS from 0.99 to 0.91 (0.85 ± 0.01 recalibrated; GRU + gap 0.82 ± 0.04, ODE-RNN 0.80 ± 0.09).
  - Widths shrink from 5.2 to 4.4 days at the same coverage.
  - On T10 the sharper intervals are more over-confident: coverage 0.19 → 0.07, MAE 5.1 → 5.5. Ambient improves
    (1.6 → 1.4 days).
- **H3 confirmed.** Recalibration lowers in-distribution CRPS 2–5 % and moves coverage to 0.86–0.93. It does
  nothing for T10. Do not apply it to the point predictors.
- Not yet rerun with guidance: E2 (exact vs MC), E3 clock ablation, E4 sparse photos.
