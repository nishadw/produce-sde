# Experiment sheet: views, time-series models and AML (avocado A2SL + mango)

2026-11-12. Follows `docs/experiments/2026-11-10_avocado-microwave-scanner-views/report.md` (rounds 1-4) and `docs/experiments/2026-11-06_mango-photo-vs-microwave/report.md`. Lab AML code read:
`../code/TMC(handwritten)` (TMC, ICLR 2021, with a distillation driver) and `../code/RepDistiller(main_steering)`
(RepDistiller with the lab's "ours" mode for steering regression).

Protocol for every row unless noted: labels are targets only (never inputs), microwave numbers z-scored, 3 seeds,
folds by fruit (avocado: 4 folds, leave one tracked fruit out; mango: 6 folds stratified by cut day), model and
hyperparameter choices on validation only, paired bootstrap over fruit, cross-fitted recalibration
(0.9 KM map + 0.1 raw). Main score: censored log score (avocado), population Brier (mango).

## What the lab code does (and how it maps onto ripen)

| lab code | mechanism | ripen today |
|---|---|---|
| RepDistiller `train_ours_partial` | each round: reset teacher from student (extra-input first-layer weights zeroed, rest copied), **head shared by reference**, teacher trained `epochs_inner_t=5`, then student 5 epochs with `gamma·task + alpha·KD + beta·other`, `gamma 0.1, alpha 0.9, T 4` | `samd` = same reset + shared predictor, but its student loss is feature MSE only (no KD term) |
| RepDistiller `train_partial` | frozen teacher, any of 13 losses: KD, FitNet, AT, SP, CC, VID, RKD, PKT, AB, FT, FSP, NST, CRD | `kd`, `feat` (≈ FitNet), `kd_feat` |
| RepDistiller partial modality | extra loss on student-only (unpaired) batches, weight `mu` | not separated: all fruit go through one loss |
| TMC model | one evidential classifier per view, Dirichlet evidence, Dempster-Shafer fusion of views | views pooled (mean+sd) or concatenated into one encoder |
| TMC `myKD(reset=True)` | teacher = all 6 views; reset zeroes every view branch and copies the student's branch in; student loss `task + alpha·MSE(fused teacher evidence, student evidence)`; grid alpha {1e-2,1e-3,1e-4} x lr {3e-2,3e-3,3e-4} x 5 repeats | no per-view branches, no grid |

Things in the lab code **not** to copy (they inflate results): `TMC/train.py` keeps the student with the best
**test** accuracy and picks the grid cell by test mean; RepDistiller's `loss_div_up` compares the student on an
unpaired batch with the teacher on a different, paired batch (mismatched fruit).

## Experiments

Status: done / new / stretch. Runs = settings x seeds x folds.

### Avocado (A2SL, 6 rotation views 0-300°)

| id | question | settings | models | status |
|---|---|---|---|---|
| AV1 | which single view is best | time, photo, mw x 6 views, photo+mw x 6 views | latent SDE | done (round 3) |
| AV1b | same, other model | as AV1 | **latent ODE** | new |
| AV2 | one model with all views | pooled 6, concat 6, pooled 0+120+240 / 60+180+300 | latent SDE | done |
| AV2b | all views, TMC-style late fusion | per-view forecasts combined: mixture and product of experts (no new training) | latent SDE, latent ODE | new, free |
| AV3 | best view setting x time-series model | best of AV1/AV2 (now photo + 0+120+240) | latent SDE, latent ODE, neural CDE, GRU-D | new |
| AV4 | AML on the best model + setting | see AML list | best of AV3 | new (kd/feat/kd_feat/samd done for latent SDE) |
| AV5 | recalibrate, diagnose, fix, rerun | all of the above | | up to 3 rounds |

### Mango (one batch of 48, destructive labels)

Mango has no rotation views. Proposed views: the 10-antenna ring gives 45 pairs; pairs at the same chord distance
see the fruit the same way, so **V1-V5 = antenna separation 1-5** (10, 10, 10, 10, 5 pairs; 5 = straight through the
centre) and **V6 = the image reconstruction**. Photos: side 1, side 2, both. Needs a re-ingest that keeps the
per-pair sweeps (today's cache averages all 45 pairs).

| id | question | settings | models | status |
|---|---|---|---|---|
| MG0 | re-ingest with per-view microwave features | V1-V6 + photo sides | | new |
| MG1 | which single view is best | mw x V1..V6, photo side 1 / 2 | latent SDE | new |
| MG2 | one model with all views | pooled, concat, late fusion | latent SDE | new |
| MG3 | best view x time-series model | best of MG1/MG2, **scan history as a time series** (every scan up to now) instead of today's one-scan snapshot | latent SDE, latent ODE, neural CDE, GRU-D | new (loader mode) |
| MG4 | AML on the best | see AML list; student = photo | best of MG3 | new (samd done, snapshot) |
| MG5 | recalibrate, diagnose, fix, rerun | | | up to 3 rounds |

### AML variations (AV4, MG4)

| id | method | from | status |
|---|---|---|---|
| A0 / A1 | photo-only baseline / teacher (ceiling) | | done (avocado) |
| A2 | CAML KD on the forecast, alpha 0.9 T 4 and alpha 0.5 T 1 | RepDistiller `kd` | done (avocado) |
| A3 | feature matching (FitNet) | RepDistiller `hint` | done (avocado) |
| A4 | SAMD: reset + shared head + feature MSE | lab "ours" | done (avocado) |
| A5 | **lab "ours" exactly**: reset + shared head + KD term, gamma 0.1 alpha 0.9 T 4 | `run_our_steering_distill*.sh` | new |
| A6-A8 | relational losses on the representation: RKD, PKT, SP (frozen teacher and with reset) | RepDistiller zoo | new |
| A9 | partial modality: separate unpaired photo-only loss, weight mu, teacher trained on scanned visits only | RepDistiller `partial` | new |
| A10 | TMC-style teacher: one branch per view + fusion; reset copies the student's photo branch in, zeroes the rest; MSE on the fused forecast | TMC `myKD` | stretch |
| A11 | small grid on the best two methods: alpha / beta {0.1, 1, 10} x lr {3e-4, 1e-3, 3e-3}, chosen on validation | TMC grid | new |

## Recommendations

1. **Late fusion before more early fusion.** TMC's point is that each view gives its own forecast plus how sure it
   is, and the fusion trusts the confident ones. It costs nothing here (per-view runs exist) and it handles the
   avocados scanned only once (missing scans) better than zero-filling.
2. **Run the lab's actual "ours" (A5).** Our SAMD drops the KD term the lab uses in every `run_our_*` script; A5
   is the closest match to their published setup and should be the headline AML row.
3. **Mango as a time series (MG3).** Today each mango forecast sees one scan. Every fruit has up to 8 scans before
   its cut day, so the neural SDE/ODE can use the trajectory; this is where a time-series model can actually beat
   the age baseline.
4. **Expect small avocado gains.** Every tracked avocado spoils in days 7-9, so differences under ~0.1 nats are
   noise. Grid search (A11) on validation only, with 3 seeds, and report the spread between identical baselines.
5. **Mango views by antenna separation** (above). Ask the lab whether the ring has 10 antennas evenly spaced; if
   not, group by the actual geometry. Ask the same about the avocado 0/120/240 split.

## Cost

Each avocado run (3 seeds x 4 folds) is ~10-20 min on one GPU or 2 CPUs; mango runs (3 x 6) similar. New runs:
~22 (AV1b) + ~8 (AV3) + ~12 (AV4) avocado and ~10 (MG1) + ~4 (MG2) + ~8 (MG3) + ~10 (MG4) mango, about 75 jobs,
roughly one night on the class partition, then recalibration rounds.
