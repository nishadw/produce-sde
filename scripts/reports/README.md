# scripts/reports

Builds the plain-language Word reports in [`results/`](../../results). Each study has a `*_figs.py` script (charts and a
numbers JSON, run in the main `ripen` environment because it reads run outputs) and a `*_report.py` script (the
`.docx`, needs only `python-docx`: `pip install -e ".[reports]"`). Charts and numbers go to `outputs/report_figures/`;
the reports read them from there. Run from the repo root, after the study's runs and summaries exist in `outputs/`.

| Report in `results/` | Build |
|---|---|
| `2026-10-28_avocado-latent-sde-vs-baselines.docx` | `avocado_fig_transfer.py`, `avocado_fig_compare.py`, `avocado_fig_heldout.py`, then `avocado_report.py` |
| `2026-11-06_mango-photo-vs-microwave.docx` | `mango_figs.py`, then `mango_report.py` (prose numbers are typed from `outputs/mango/summary.md`) |
| `2026-11-10_avocado-microwave-scanner-views.docx` | `a2sl_views_figs.py`, then `a2sl_views_report.py` |
| `2026-11-12_views-time-series-models-and-aml.docx` | `views_aml_figs.py`, then `views_aml_report.py` |
| `2026-11-13_results-brief-ode-sde-aml.docx` | `views_aml_figs.py`, `brief_figs.py`, then `brief_report.py` |

Example:

```bash
python scripts/reports/brief_figs.py      # outputs/report_figures/br_*.png + brief_numbers.json
python scripts/reports/brief_report.py    # results/2026-11-13_results-brief-ode-sde-aml.docx
```
