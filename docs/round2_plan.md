# Round 2: improved model (avocado, E1–E4 again)

Started 2026-11-04. Round 1 = `results/avocado_seeds/` (5 seeds, fixed split). Round 2 changes three things,
each aimed at a weakness Round 1 found, and keeps everything else (split, seeds 1–5, epochs, configs) identical.

| Change | Fixes | Applies to | How |
|---|---|---|---|
| **DINOv2 ViT-B/14 photo features** (CLS token, z-scored over all photos, unsupervised) instead of frozen ImageNet ResNet-18 | In-distribution the latent SDE (CRPS 0.99) loses to ODE-RNN / GRU + gap (0.83–0.84); ripeness is colour and texture, which supervised ImageNet pooling partly discards | every model (fair comparison) | `data.features_file=data/features/avocado_dinov2_vitb14.npz` |
| **Guided posterior** (closed-form h-transform toward the next label) + `event_mean=log_mean` | E0: the unguided ELBO biases clock rates 14–24 % fast; fixed on synthetic (+2–10 %) but never used on real data | latent SDE runs | `model.posterior_guidance=true model.event_mean=log_mean` |
| **Post-hoc PIT recalibration** (Kaplan–Meier on the PIT scale, fit on val, applied to test) | Over-wide intervals in-distribution (cov. 0.93–0.96) | every run, Round 1 and Round 2 (no retraining) | `scripts/recalibrate.py` → `metrics_test_recal.json` |

Ablations (latent SDE, in-distribution + 3 LOGO folds): guidance alone (`*_guided_r18`) and DINOv2 alone
(`*_dino_unguided`), so the Round 1 → Round 2 change can be split between the two.

## Gate before the full run
1. `bash run_umiacs.sh features2`: extract the cache, then a linear probe (`results/feature_probe.md`): ridge from one
   visit's features to stage and days-left, train → val. Go only if DINOv2 is at least as good as ResNet-18.
2. `bash run_umiacs.sh r2-pilot`: one seed of the round-2 latent SDE and GRU-D, to catch errors.
3. `bash run_umiacs.sh r2`: 45 runs × 5 seeds → `results/avocado_round2/summary.md` and `summary_recal.md`.

## Hypotheses (written before the results)
- H1: DINOv2 lowers in-distribution CRPS for every image model; the latent SDE closes most of the gap to ODE-RNN.
- H2: Guidance sharpens the latent SDE's intervals (Round 1 width 5.2 d) without losing coverage, and slows the fitted clock.
- H3: Recalibration brings in-distribution coverage to the oracle level and lowers ECE, but does not fix T10
  (the validation PIT comes from the training temperatures).
- H4: The Arrhenius-vs-ablated-clock result (E3) survives the new features.

## Note on the 80 % coverage target
`compute_metrics` takes quantiles on the integer-day grid and counts `q10 <= hi <= q90` inclusively, so a forecaster
that knows the true law scores **0.85–0.88**, not 0.80 (measured in `tests/test_recalibrate.py` style simulations with
true sd 0.7–3 days). Round 1's 0.93–0.96 is over-coverage, but by less than it looks; the latent ODE's 0.64 is worse
than it looks. Compare coverage against that oracle level, or use ECE.

## Results (5 November; details in report/REPORT.md, "Round 2")
- Gate: probe stage accuracy 0.759 (DINOv2) vs 0.714 (ResNet-18); days-left MAE 2.96 vs 2.91. Pilot clean.
- **H1 rejected.** DINOv2 made every sequence model worse:
  - GRU + gap: CRPS 0.84 → 0.96
  - GRU-D: 0.90 → 0.99
  - FPT filter: 0.89 → 0.92
  - latent SDE, unguided: 0.99 → 1.01

  It helped only the per-visit models (discrete survival 1.08 → 0.98, direct regressor 1.66 → 1.56). The
  single-visit probe was the wrong gate.
- **H2 confirmed in-distribution, not under transfer.**
  - Guidance takes CRPS from 0.99 to 0.91 (ResNet-18) and to 0.90 (DINOv2).
  - Widths shrink from 5.2 to 4.2–4.4 days at the same coverage.
  - On T10 the sharper intervals are more over-confident: coverage 0.19 → 0.07, MAE 5.1 → 5.5.
- **H3 confirmed.** Recalibration lowers in-distribution CRPS 2–5 % and moves coverage to 0.86–0.93. It does
  nothing for T10. Do not apply it to the point predictors.
- **H4 confirmed.** Arrhenius beats both ablated clocks in 15/15 seed × group pairs (1.4–2.3×).
- **Not predicted:** with guidance the exact PDE likelihood beats MC (0.90 vs 0.95), reversing E2.
- **Best configuration:** guided posterior + ResNet-18 + recalibration, CRPS 0.85 ± 0.01. GRU + gap scores
  0.82 ± 0.04 and ODE-RNN 0.80 ± 0.09 (both Round 1, recalibrated).
