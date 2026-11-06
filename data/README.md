# data/

Raw data is never committed. Only this file, `manifests/` and the split files are tracked.

```
data/
  avocado/     Mendeley 'Hass' avocado set (10.17632/3xd9n945v8.1, CC BY 4.0): spreadsheet + images/*.jpg
               (python scripts/download_avocado.py, resumable)
  a2sl/        A2SL microwave scans + photos (scripts/ingest_a2sl.py; see docs/data_a2sl.md)
  mango/       Spring 2026 mango study (48 fruit, one batch): "Mango data classification.xlsx" (destructive
               cut-day measurements), "Mango pictures/" (2 photos per fruit, cut day), "Spring 2026/" (microwave
               scans every other day: 45 .s2p sweeps + a 72x72 reconstruction per fruit and day, plus an air scan).
               scripts/ingest_mango.py -> manifests/mango_*.csv and mango/mango_scans.npz
  features/    frozen-encoder features from scripts/extract_features.py
```

Other candidate datasets: `docs/dataset_survey.md`.
