# Mango study: days until spoiled from photos, microwave scans, and photo-only distillation

**6 November 2026** · data: `data/mango/` (Spring 2026, one batch of 48 mangoes) · code: `ripen.data.mango`,
`ripen.train.samd`, `ripen.eval.mango`, `scripts/mango_cv.py` · numbers: [`results/mango/summary.md`](../results/mango/summary.md)
(6-fold cross-validation by fruit × 3 seeds; reproduce with `bash run_umiacs.sh mango`).

> **Summary.**
> - **Microwave scans predict much better than photos.** For when a mango will spoil, every neural model does
>   clearly better with microwave scans than with one photo (population Brier ≈ 0.15 vs ≈ 0.20; significant in a
>   paired bootstrap). Microwave also reads firmness and ripeness class better.
> - **Distillation mostly does not rescue the photo-only model.** Distilling the microwave knowledge into a
>   photo-only model (SAMD, Shen et al. ICML 2023) does **not** help the standard neural models. It helps the
>   structured ones (latent ODE, FPT filter) only because their own photo-only versions were poor.
> - **A two-line baseline beats every neural model.** The baseline estimates the fruit's age from the sensor, then
>   reads the batch's spoilage curve. No model, sensor or setting can tell apart fruit of the *same* age.
> - **The data limit is the main finding.** In this dataset, "days until spoiled" is mostly "how old is this fruit".

---

## 1. The data and what it can answer

| | |
|---|---|
| Fruit | 48 labelled mangoes, bought together and stored together (temperature not recorded); 3 more were scanned but never cut |
| Design | **destructive**: every other day (study days 0, 2, …, 14) six fruit were cut open and measured |
| On the cut day | hue angle, firmness (N), sweetness (°Brix), ripeness class from firmness (Unripe > 30 N, Underripe 15–30, Ripe 3–15, Overripe 1–3, **Senescent ≤ 1 N**), and 2 photos (3 fruit have 1) |
| Before the cut | a microwave scan of every remaining fruit every other day: 45 antenna-pair S-parameter sweeps (3.3–8 GHz, 201 points) and a 72×72 image reconstruction, plus an empty-scanner ("air") scan each day |

Average per cut day (6 fruit each):

| day | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|
| firmness (N) | 38.3 | 20.4 | 7.7 | 3.6 | 2.6 | 1.4 | 1.1 | 1.5 |
| hue (°) | 84.1 | 85.0 | 83.1 | 82.1 | 80.9 | 80.5 | 79.1 | 73.8 |
| senescent | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 3/6 | 6/6 |

**Spoiled = senescent.** It appears only on days 12 and 14. Each fruit is seen spoiled-or-not on **one** day, so this
is *current-status* data: no fruit has an observed spoilage day. Two consequences shape everything below:

1. **Labels.** All 30 fruit cut on days 0–10 were not senescent, so we assume the batch did not start spoiling before
   day 10. A fruit senescent on day 12 or 14 then spoiled in (10, 12] or (10, 14]; one not senescent when cut on day
   *d* is right-censored at *d*. Relative to a scan on day *s* these become intervals of *days from now*. The
   microwave scans before the cut therefore carry real "how many days" information. The photo, taken on the cut day,
   carries only "spoiled now or not".
2. **Scoring.** There is no per-fruit "true days left" to compare with. What the data can check:
   - **Population Brier (main score).** Fruit were cut on effectively random days, so the fruit cut later in the same
     test fold show how much of the batch had spoiled by then. For fruit *i* (scored on its cut day *d_i*) and every
     fold-mate *j* cut on day *d_j ≥ d_i*, compare *P(i spoils within d_j − d_i days)* with *j*'s status. This checks
     the "how many days" part of the forecast. Only test-fold fruit are used, so no training label enters the score.
   - **Spoiled-now Brier**: the fruit's own status on its cut day.
   - **Firmness rank correlation**: expected days left vs measured firmness (firmer → longer). The **within-day**
     version removes each day's mean first, so it only credits telling apart fruit *of the same age*.
   - **Earlier-scan log score** (microwave settings only): the proper interval-censored score at every scan before the cut.

Every model and setting is scored on the same 48 fruit at the same moment: the cut day, with no stage labels visible.

## 2. Inputs, models and distillation

**Snapshot instances.** A prediction is made from one visit "now" (no age information): the cut day for a photo, any
scan day for the microwave. Microwave features are the current scan and its change since day 0. Settings:

| setting | input at prediction time | training instances |
|---|---|---|
| photo only | ResNet-18 features of the cut-day photos (16 principal components) | 48 fruit × 1 (cut day) |
| microwave only | microwave scan now + change since day 0 (16 + 16 components) | every scan (216) |
| photo + microwave | `[photo | photo-present flag | microwave]` | every scan; photo on the cut day only |
| photo only, SAMD | photo (as photo only); trained with a photo + microwave teacher | as both |

Principal components are fitted on all fruit without labels; folds are by fruit (every scan of a fruit stays on one side).

**SAMD** (Smart Auxiliary Modality Distillation; Shen, Wang, Gao & Lin, ICML 2023, App. A.7; no public code, so
implemented from the paper in [`ripen/train/samd.py`](../src/ripen/train/samd.py)):
- The teacher (photo + microwave) is a **supermodel** of the student (photo): same network, with extra input columns.
- Each round:
  1. **Reset** the teacher from the student, with the microwave columns set to zero. It then computes exactly the
     student's function, which is tested in `tests/test_mango.py` for four model types.
  2. Train the teacher for *k* = 5 epochs.
  3. **Share** its predictor (everything after the encoder) with the student.
  4. Train the student for *k* epochs on its own loss + β·‖F_student − F_teacher‖² (feature consistency, β = 1).
- 12 rounds, with early stopping on the student's validation loss.
- The other papers in the request are about robustness to image corruptions (sensitivity-guided augmentation;
  gradient-free adversarial training). They were not used, because 48 fixed-lighting studio photos give no
  corruption to be robust to.

**Models**: every learned model in `ripen.models` (classical Wiener / gamma / Cox need a stage history as input,
which a photo user does not have).
- **On one visit, GRU + gap, GRU-D and ODE-RNN are the same function.** With no time gap there is nothing for GRU-D's
  decay or the ODE to act on, so the three give identical numbers (shown once below).
- **The latent SDE and latent ODE** are the main model family, unguided (the guided posterior reads labels at
  prediction time). They use the stage labels as training targets.
- **The FPT filter** is the exact grid filter.

**References** (closed form, no network): fit the batch's spoilage curve *P(spoiled by day t)* from the training
fruit's cut-day status (isotonic regression), then forecast from the fruit's age. The age is either **known**
(an oracle, since a shopper does not know it) or **estimated** from the sensor by ridge regression.

## 3. Results

Full tables (every score, every setting): [`results/mango/summary.md`](../results/mango/summary.md). GRU + gap,
GRU-D and ODE-RNN are identical on single visits and are shown as one row.

**Main score: population Brier of the days-until-spoiled forecast** (lower is better; mean ± std over 3 seeds)

| model | photo only | photo only, SAMD | microwave only | photo + microwave |
|---|---|---|---|---|
| GRU / GRU-D / ODE-RNN | 0.200 ± 0.009 | 0.213 ± 0.019 | **0.147** ± 0.004 | 0.151 ± 0.006 |
| neural CDE | 0.197 ± 0.003 | 0.214 ± 0.010 | 0.154 ± 0.005 | **0.143** ± 0.014 |
| transformer | 0.199 ± 0.011 | 0.203 ± 0.011 | 0.152 ± 0.004 | 0.155 ± 0.011 |
| latent SDE | 0.262 ± 0.005 | 0.229 ± 0.020 | 0.177 ± 0.003 | 0.173 ± 0.007 |
| latent ODE | 0.271 ± 0.012 | 0.226 ± 0.014 | 0.175 ± 0.017 | 0.184 ± 0.013 |
| FPT filter | 0.329 ± 0.003 | 0.213 ± 0.004 | 0.212 ± 0.001 | 0.212 ± 0.001 |

**References** (no neural network)

| reference | population Brier | spoiled-now Brier | Spearman vs firmness | within-day Spearman |
|---|---|---|---|---|
| known age + batch curve (oracle) | **0.077** | **0.045** | **0.93** | −0.22 |
| photo → estimated age + batch curve | 0.182 | 0.136 | 0.84 | −0.10 |
| microwave → estimated age + batch curve | 0.137 | 0.171 | 0.78 | −0.31 |
| photo + microwave → estimated age + batch curve | **0.118** | 0.161 | 0.82 | −0.29 |

**Paired bootstrap** (fruit resampled; difference in population Brier, 95 % interval; `*` = interval excludes 0;
negative = first is better)

| model | SAMD − photo only | photo + microwave − photo only | SAMD − photo→age reference |
|---|---|---|---|
| GRU / GRU-D / ODE-RNN | +0.013 [−0.013, +0.040] | −0.049 [−0.078, −0.018] * | +0.030 [+0.002, +0.063] * |
| neural CDE | +0.017 [−0.014, +0.055] | −0.054 [−0.080, −0.026] * | +0.031 [−0.002, +0.069] |
| transformer | +0.004 [−0.009, +0.019] | −0.044 [−0.074, −0.011] * | +0.021 [−0.008, +0.054] |
| latent SDE | −0.033 [−0.095, +0.016] | −0.089 [−0.143, −0.041] * | +0.046 [+0.027, +0.067] * |
| latent ODE | −0.045 [−0.089, −0.003] * | −0.087 [−0.141, −0.035] * | +0.044 [+0.003, +0.098] * |
| FPT filter | −0.116 [−0.168, −0.052] * | −0.117 [−0.177, −0.044] * | +0.031 [−0.003, +0.076] |

**Other outputs on the cut day** (ridge read-out from the same features)

| features | firmness error (N) | ripeness class exact | class within one |
|---|---|---|---|
| photo | 6.6 | 38 % | 81 % |
| microwave | 5.7 | 44 % | 94 % |
| photo + microwave | **5.6** | **52 %** | **94 %** |

Other columns of `summary.md`:
- **Firmness rank correlation** (expected days left vs firmness): 0.61–0.76 for the neural models with microwave
  input, 0.0–0.27 with photo only.
- **Within-day rank correlation**: between −0.37 and +0.14 everywhere, so none of them is clearly positive.
- **Mean predicted spoilage day**: 14.2–15.2 with microwave input, 15–20 with photo only (true: 10–14).

## 4. What this means

- **Microwave beats photo for timing, by a clear margin.** It is the sensor that tracks the fruit over time.
- **Photo-only forecasts are weak.** A single cut-day photo carries little timing information beyond "spoiled now?".
  SAMD transfers some of the teacher's timing to structured models, whose photo-only versions were badly calibrated.
  It does not improve the standard neural students, and no distilled student beats the two-line "photo → age → batch
  curve" baseline.
- **Age is almost everything here.** Knowing the fruit's age alone gives population Brier 0.077, far better than any
  sensor model. Within a day, no model or sensor ranks fruit by firmness better than chance. With 6 fruit per day and
  ripening this synchronised, per-fruit differences are small next to measurement noise.
- **The neural models forecast too late.** They put the average spoilage day at 14–16. The batch spoiled between days
  10 and 14, and the references put it at 12.5–13.8. With 32 training fruit, a censored likelihood that rarely sees
  a spoiled fruit pushes mass late.

**Caveats.**
- 48 fruit, one batch, one storage condition (temperature not recorded).
- The "no spoilage before day 10" assumption.
- Principal components fitted on all fruit (label-free).
- Marker writing on the fruit is visible in the photos (identical circles on all fruit; IDs unrelated to cut day).
