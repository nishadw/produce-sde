# A2SL: all 15 avocados, real photos only (2026-11-19)

Replaces the half-synthetic set as the main study (user, 2026-11-19: "use all the fruit in this dataset ... real photos
only, drop fixed start"). The half-synthetic results stay in `2026-11-17_a2sl-imputation-random-start.md`.

## Data
- `load_a2sl_avocado` as is: every fruit's real visits, day-rule labels (day 9 = spoiled), visits after the first
  spoiled one dropped. No fill (`--synth real`).
- Clock: unchanged. Each sequence starts at t = 0 at its first visible photo; random start already re-zeroes, so a fruit
  first photographed on day 7 (avo12) is the same kind of example as a random start.
- What each fruit gives:

| Fruit | Real photos | Outcome |
|---|---|---|
| avo01, 02, 03, 08, 09, 10, 11 | day 1 to day 9-13 | spoiled in (7, 9] (observed) |
| avo12 | days 7, 9 | spoiled in (7, 9] (observed) |
| avo07 / 06 / 05 / 04 | to day 7 / 5 / 3 / 1 (cut) | not spoiled by then (censored) |
| avo13, 14, 15 | day 9 only | already spoiled at the only photo: no forecast to make |

## Folds
`configs/splits/a2sl_all_fold{0..11}.json` (`a2sl_views.write_all_folds`): test = one of avo01-12; val = two observed
fruit other than the test fruit; train = the other 12 (avo13-15 always train). Matching leaves avo13-15 out of its
reference set (no unspoiled moment to match against).

## Runs (`scripts/run/submit_all.sh`)
Random start only, 5 seeds:
- grid (`outputs/a2sl_all`): matching, Weibull, latent ODE, latent SDE, transformer x photo / date only;
- photo-only students (`outputs/a2sl_all_aml`): latent ODE / SDE / transformer x teacher, CAML-KD, SAMD, each with a
  photo + scanner teacher and with a photo-only teacher.

## Scoring (`scripts/analysis/a2sl_all_compare.py`)
Every held-out fruit from every start day. Headline on the 8 observed fruit (the 4 cut fruit only say "not spoiled
yet", which any forecast gets right); all 12 shown too. Paired test vs the no-photo forecast with a bootstrap over
the 8 fruit.

## Caveat
Labels are the day rule, so every fruit spoils in (7, 9]: more fruit give more test cases, not fruit-to-fruit variation.

## Results (2026-11-19, 109 runs, all completed)

Full table: `outputs/a2sl_all_compare/summary.md`; brief: `../results/2026-11-19_all-avocados-real-photos.docx`.
Score on the 8 observed fruit (no-photo forecast 1.15; all 12 fruit: 0.98).

- **Nothing is clearly better than the no-photo forecast.** Closest: image matching 1.23 (no clear difference;
  better on 6 of 8 fruit but +1.82 on avo08 and +0.82 on avo11; lowest day error, 0.9 d vs 1.5 d).
- Weibull 1.41 photo / 1.30 date only. Transformer 1.34 photo (worse, 3 of 8), latent SDE 1.58, latent ODE 2.75.
- Teachers (CAML-KD, SAMD) never beat their photo-only model by enough to matter; the best are latent SDE + SAMD 1.52
  and transformer + CAML-KD 1.51. Transformer + SAMD trained this time (1.79) but is still worse.
- Blend chosen on validation: Weibull photo reaches 1.10 (untested); every other model's mix goes to w ≈ 1.
  The test-scored curve for the transformer (photo) dips to 1.11 at w = 0.7, which validation did not find.

## Follow-up (2026-11-21)
- Teacher-alone rows dropped from the table (they need the scanner at test time).
- Image matching blend chosen on validation: w = 0 (model alone, 1.23). The test-scored curve dips to 0.96 at w = 0.4,
  but 2 validation fruit per fold did not find it.
- New `storage_day` model (`ripen.models.classical.StorageDay`, `scripts/run/submit_storage_day.sh`): photo -> storage
  day (ridge multinomial logistic on 8 PCA components, combined over the visible photos via their gaps), then the
  training fruit's spoilage interval in storage days. **0.33 vs 1.15, better on 8 of 8 fruit (bootstrap excludes 0),
  day error 0.6 d.** Date only: 1.17. Caveat: every fruit shared one photo session per day, so a session shortcut
  (lighting/background) cannot be ruled out with this data; the skin colour change is visible by eye.

## Improvements (2026-11-22)
User: storage day "lowk already knows the answer" (it trains on each photo's storage day, which the day rule makes the
label), so it is out of every comparison. Tried instead (`outputs/a2sl_improve/summary.md`,
`scripts/analysis/a2sl_improve_compare.py`, runs from `scripts/run/submit_improve.sh`):
1. + 2. Temperature T and blend w with the no-photo forecast chosen on the other 11 held-out fruit (not 2 val fruit).
3. Background removed (`ripen.data.fruit_mask`): fruit-only ResNet-18, and 12 skin-colour numbers from fruit pixels.
4. Ripeness reader trained on the 478 public avocados' Ripening Index (fruit-only ResNet-18, 69 % held-out accuracy);
   on the lab photos it saturates by day 3 (expected stage 2.4 on day 1, ~4.3 from day 3 on).

Score on the 8 observed fruit (no-photo 1.15):
- Image matching: original 1.23 raw / 0.92 recalibrated; fruit only 0.83 / 0.86 (better); **skin colour 0.28 raw,
  better on 8 of 8, day error 0.4 d**; public reader 0.79 / 0.86.
- Transformer: recalibrated 1.07-1.14 with any features (original and public reader significantly better than 1.15,
  by 0.05-0.08). Latent SDE / ODE: recalibration picks the no-photo forecast alone (w = 1) for every feature set.
- Caveat: colour is measured under each day's session lighting; background removal does not remove that.
