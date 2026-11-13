# Experiments

One folder per study, named `<date first run>_<what it tested>`, so they sort chronologically. Each holds the technical
write-up (`report.md`), the summary tables it cites (`tables/`) and, where used, figures. The plain-language Word
version of each study is in [`results/`](../../results) under the same name.

| Date | Study | What was done | Headline |
|---|---|---|---|
| 2026-10-28 | [Avocado: latent SDE vs baselines](2026-10-28_avocado-latent-sde-vs-baselines/report.md) | Public Mendeley set (478 avocados, daily photos, 3 storage temperatures). Latent SDE with an exact first-passage likelihood against 12 baselines; transfer to an unseen temperature, sparse photos, likelihood ablations; 5 seeds, paired bootstrap. Rounds 1–2b ran through 2026-11-05. | Best at transfer and sparse photos (Arrhenius clock cuts held-out MAE 1.5–2.4×); a small transformer wins in-distribution |
| 2026-11-06 | [Mango: photo vs microwave](2026-11-06_mango-photo-vs-microwave/report.md) | 48 lab mangoes, one batch, cut open on days 0–14. Every model × {photo, microwave, both, photo model distilled from both with SAMD}. | Microwave clearly beats photo; nothing beats a known-age reference |
| 2026-11-10 | [Avocado: microwave scanner views](2026-11-10_avocado-microwave-scanner-views/report.md) | 15 lab avocados (4 tracked). One model per scanner rotation view (0°–300°) and pooled subsets; first teacher-student (AML) runs: CAML, feature matching, SAMD. | Views 0°/120°/240° beat 60°/180°/300°; no photo-only student beats photo-only |
| 2026-11-12 | [Views × time-series models × AML](2026-11-12_views-time-series-models-and-aml/report.md) | Both fruits. Lab's TMC (late fusion) and RepDistiller (KD/RKD/PKT/SP, reset + shared head) methods ported; latent SDE vs latent ODE; controls for the training schedule. [Experiment sheet](2026-11-12_views-time-series-models-and-aml/experiment_sheet.md). | Latent ODE best on both fruits; reset-style teaching gains come from the schedule, not the scanner |
| 2026-11-13 | [Results brief: ODE, SDE, AML](2026-11-13_results-brief-ode-sde-aml/README.md) | Training-schedule ladder (5 seeds × 4 folds), CAML / lab method / teacher reruns at 5 seeds, mango corrections fitted to the mango score. | Scanner barely helps even the teacher; gains are from training rounds; mango forecasts stay uncorrected |
