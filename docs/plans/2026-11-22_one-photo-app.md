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
