# One-page results summary (Word)

`../Produce_Spoilage_SDE_Results.docx` is the plain-language summary of the main model (guided latent SDE,
ResNet-18 features) for presenting. Rebuild it after new results:

```bash
python report/brief/make_transfer_figure.py   # transfer.png, from results/avocado_seeds (needs matplotlib)
python report/brief/make_compare_figure.py    # compare.png, from results/avocado_seeds + avocado_round2
python report/brief/make_docx.py              # report/Produce_Spoilage_SDE_Results.docx (pip install -e ".[brief]")
```

Numbers in the docx text and table are typed in `make_docx.py`; update them there when results change.
