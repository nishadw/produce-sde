# One photo in, days until spoiled out (2026-11-22)

**Goal.** The end product is an app: the user photographs an avocado and gets back how many days until it spoils.
So every model sees one photo and nothing else: no date, no storage day, no storage group or temperature, no
ripeness-stage label. No date-only forecast and no blending toward one. Recalibration is parked until this phase is in.

**Phase 1 (this plan).**
- Data: `ripen.data.schema.one_photo` turns every unspoiled visit into its own item (`<fruit>@<k>`, one visit at
  t = 0, spoil window in days from the photo); storage/temperature are set to one constant; test items have stage
  labels hidden. Splits stay by fruit.
- Train on the public avocados (`configs/splits/avocado_random_seed0.json`: 334/71/73 fruit, about 4,100/900/900
  photos). Test on the held-out public fruit and on the lab avocados (42 unspoiled photos of 12 fruit; never train).
- Photo variants: full photo (ResNet-18), fruit only (ResNet-18 of the cut-out), skin colour (12 numbers, z-scored
  on the public photos for both sets): `scripts/data/one_photo_features.py`.
- Models: image matching (`k_visits=1`), transformer (with one photo this is a direct photo -> days network,
  `group_dim=0`), latent SDE and latent ODE (`use_stages=false` as input; the SDE still trains on stage labels).
- Runs: `scripts/run/submit_one_photo.sh` (matching once, others 3 seeds) -> `outputs/one_photo/`; summary
  `scripts/analysis/one_photo_compare.py`.
- Scores: censored log score on the days-left window, day error, 80 % range coverage; averaged per fruit.

**Phase 2 (next).** Fine-tune on the lab avocados and add the microwave scans as a training-only teacher
(CAML-KD, SAMD): the app never needs a scanner. Colour correction against the background. Recalibration revisited.

## Phase 1 results (2026-11-23)

All 31 jobs completed (`outputs/one_photo/summary.md`). Fruit-weighted, mean over seeds; public test 73 fruit / 905
photos, lab 15 fruit / 42 photos (3 photographed already spoiled).

| Model / photo | Public score | Public day error | Lab score | Lab day error | Lab 80 % range |
|---|---|---|---|---|---|
| Matching, skin colour | 1.90 | 2.2 d | 3.96 | 2.4 d | 48 % |
| Transformer, skin colour | 1.80 | 2.4 d | 3.27 | 9.1 d | 52 % |
| Transformer, fruit only | 1.88 | 2.7 d | 3.56 | 2.1 d | 45 % |
| Latent SDE, fruit only | 2.23 | 2.8 d | 2.56 | 1.9 d | 73 % |
| Latent SDE, skin colour | 2.27 | 2.8 d | 1.84 | 2.4 d | 80 % |
| Latent ODE, fruit only | 2.77 | 3.0 d | 2.41 | 2.1 d | 63 % |

- Public: every model 2.2-3.2 days off; skin colour best for transformer and matching.
- Background removal barely matters on public photos (white background) but lowers lab day error for every model.
- Latent SDE transfers best to the lab; transformer on colour breaks (lab skin ~2 SD darker than the public mean).
- Lab avocados go black by day 3 and last 4-6 more days; public-trained models call them nearly spoiled.
- 10 °C fruit hardest (2.4-4.0 d vs 1.2-2.8 d at 20 °C). SDE ranges too wide on public (98 %).
- Lab seed spread is large (e.g. latent SDE colour lab score 1.52-2.38); 15 lab fruit only.

## Phase 2: microwave teacher (submitted 2026-11-23)

- `scripts/run/one_photo_teacher.py`: start from each phase-1 public model; fine-tune on lab avocados, leave one of
  avo01-12 out (val: two of avo08/09/10/12, never scanned, so every scanned fruit trains; avo13-15 always train).
- Only 20 lab photos have scans (avo01/02/03/11 every visit; one each for avo04-07). Teacher terms on those only.
- Methods: `finetune` (photo, no teacher: the control), `teacher` (public model widened to photo + scan, zero-init
  scan inputs; needs scanner), `kd` (CAML 0.1 task + 0.9 KD, t 4), `kd_photo` (control: photo-only teacher), `samd`.
- Models latent SDE / transformer / latent ODE x fruit-only / skin colour x 3 seeds: jobs 8540149-8540166, summary
  8540167 -> `outputs/one_photo_teacher/summary.md` (`scripts/analysis/one_photo_teacher_compare.py`).
- Mangoes not used yet (the student is an avocado app).
