# Plan: fill in the missing A2SL visits, then train with a random start (2026-11-17)

Models from here on: latent ODE, latent SDE, CAML-style KD and SAMD (both distil into a photo-only latent ODE/SDE),
and later the transformer. Status: **plan, nothing run yet.** Open questions are at the end.

## 0. What the data actually looks like

15 lab avocados, a visit every other day (days 1, 3, ..., 13).

| Fruit | Photos on | Scans on | Why it stops |
|---|---|---|---|
| avo01, 02, 03, 11 | all 7 visits | all 7 visits | followed to the end |
| avo04 … avo10 | day 1 up to the cut day (1, 3, 5, 7, 9, 11, 13) | **cut day only** | cut open (destructive check) |
| avo12 | days 7, 9 | none | joined late |
| avo13, 14, 15 | day 9 | none | joined late |

The data is a **staircase**, not a shared window. All 11 fruit that start on day 1 overlap only on day 1, and 9 of
them overlap on days 1–5. The scans have no shared window at all: the cut fruit have a single scan each, on
different days.

The 6-fruit pilot (May 2023) adds daily photos (4 side views), cut on days 1, 4, 7, 10, 13 and 13. Two pilot fruit
have daily photos all the way to day 13. It is not used today.

**Labels are set by the day, not by the fruit.** Every visit gets the published class for its study day (Wu et al.
2026), so every fruit "spoils" in (day 7, day 9] by construction. This matters for everything below: see §4 and
question 2.

## 1. Research: how to fill in missing visits

Simplest first. Only steps A–C are worth running at n = 15. D is listed so we can say why we skip it.

| | Method | Fits this data? | Reference |
|---|---|---|---|
| A | Per-feature linear trend in day: a slope shared by all fruit plus each fruit's own offset (random-intercept linear mixed model) | yes; uses every observed (fruit, day), staircase or not | Laird & Ware 1982 (Biometrics) |
| B | Logistic (S-curve) trend per feature with a shared shape and a per-fruit time shift | yes; ripening colour and firmness follow S-curves, and the per-fruit shift is exactly a "biological age" | standard fruit-ripening growth curves; same mixed-model fitting |
| C | Multiple imputation (MICE) / Gaussian-process draws: several filled-in copies, each with realistic noise | yes, as a noise layer on A or B | van Buuren & Groothuis-Oudshoorn 2011 (J. Stat. Softw.) |
| D | Learned imputers: BRITS, SAITS, CSDI, GP-VAE (PyPOTS has all four) | no; they need hundreds of series | Cao et al. 2018; Du et al. 2023; Tashiro et al. 2021; Fortuin et al. 2020 |

Also worth stating in the write-up: the latent ODE/SDE already train on ragged, irregular series (Rubanova et
al. 2019), so filling in is not needed to make them run. What filling in adds is full-length training trajectories
for 7 more fruit. It adds **no new information**: the filled-in values come from a model of the same 15 fruit.

## 2. Step by step

**Step 0. Clean up** (see question 1 for scope). Remove code, configs and tests for the models we no longer use.
Keep their past result write-ups (git history keeps the code).

**Step 1. Features to fill in.** Use photo ResNet-18 features (512-d), reduced to k = 8 principal components fit on
the training fruit of each fold. Use scans as the 0°/120°/240° summaries. Fill in the components, not the raw 512-d
vector.

**Step 2. Imputer, following your truncation rule.**
- Choose a window W where every fruit used is observed: days 1–5, with 9 fruit (avo01–03, 06–11).
- Fit A (linear) and B (logistic) on W only, then extrapolate each fruit to day 13.
- Scans: fit the shared trend on the tracked fruit only, then anchor each cut fruit to its single scan (offset only).
- The imputer is refit inside every fold, without the held-out fruit, so nothing leaks.

**Step 3. Check the extrapolation before using it.**
- The 4 tracked fruit have real days 7–13. Fit on days 1–5, predict 7–13, and compare with the real features.
- Error per component, against two naive fills: carry the last value forward, and the population mean per day.
- If A and B don't beat carry-forward, the synthetic data is noise, and we say so and stop at step 5 with real data only.

**Step 4. Build the synthetic, complete set.**
- 11 fruit × days 1–13. avo12–15 are left out because they start late and would need backward extrapolation.
- Every visit carries an `imputed` flag.
- Add residual noise resampled from the fit and draw M = 5 copies (C), so the filled-in curves aren't unrealistically smooth.
- Labels for filled-in visits come from the same day rule.

**Step 5. Random start.**
- Training: each epoch, for each fruit, draw a start visit s ∈ {1, 3, 5, 7} and keep visits from s onward.
- Time is re-zeroed (t′ = day − s), so the model never sees the absolute day.
- The target is the shifted spoilage window (7 − s, 9 − s].
- Test: score the held-out tracked fruit from every start day and report the score by start.
- Built-in check: under random start the date-only model should fall to the population average. Any image model
  that beats it is reading ripeness from the photo, not from the calendar.

**Step 6. Runs.** Leave-one-tracked-fruit-out (4 folds) × 5 seeds. Scores use **real visits only**; a filled-in
visit is never scored.

| Factor | Levels |
|---|---|
| Training data | real only · real + linear fill · real + logistic fill |
| Start | fixed (day 1, as now) · random |
| Model | latent ODE · latent SDE (photo only; date only as the reference) |

That is 3 × 2 × 2 × 20 = 240 short jobs. On the best cell, rerun CAML-style KD and SAMD (photo-only student, teacher
also sees scans) against photo only: 3 methods × 20 = 60 more. Transformer later, same grid.

**Step 7. Report.** A short docx in the brief style:
- extrapolation check (step 3),
- the 3 × 2 table per model,
- score by start day,
- the AML table.

## 3. Metrics

- **Censored log score** as before.
- **Day error**: the error of the median forecast.
- Under random start, also the **age error**: how far the model's implied storage day is from the true one. This
  says directly whether the photo carries age information.

## 4. Risks to say up front

- **Day-set labels.** With them, random start turns the task into "estimate the fruit's age from its photo". The
  spoilage day itself never varies, so neither filling in nor any model can learn fruit-to-fruit differences.
  Fixing this needs per-fruit labels (question 2).
- **Too few fruit.** Four test fruit still limit what we can tell apart (about 0.1 in log score). Filling in grows
  the training set, not the test set.
- **Extrapolation past day 5** is a guess about curve shape. Step 3 measures how good the guess is.

## 5. Open questions

1. Cleanup scope (models only, or also the mango / watermelon / 478-fruit Kaggle pipelines).
2. Labels: keep the day rule, or label the 7 cut photos (and the pilot cut photos) per fruit with the lab?
3. Use the pilot set (2 more fruit with daily photos to day 13)?
4. Random start on training only, or also report by start day at test (plan assumes both)?
