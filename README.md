# Predicting When Produce Spoils with Latent Neural SDEs

[![tests](https://github.com/nishadw/produce-sde/actions/workflows/tests.yml/badge.svg)](https://github.com/nishadw/produce-sde/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-1F2328)
![jax](https://img.shields.io/badge/JAX-Diffrax%20%7C%20Equinox-C8102E)
[![license](https://img.shields.io/badge/license-MIT-6B7280)](LICENSE)

Given a few photos (and, for some fruit, microwave scans) taken at irregular times, forecast a **calibrated
distribution over the day a fruit spoils**. The core model learns a latent Itô diffusion for ripeness and treats
spoilage as the first time it crosses a boundary. The likelihood of interval-censored spoilage labels is **exact**:
it comes from the Kolmogorov PDEs, not from path simulation. Built in JAX / Equinox / Diffrax, with Hydra configs
and SLURM drivers.

## Methods

| Piece | Tool |
|---|---|
| Latent ripeness `dR = κ(x) μ_θ(R) dt + √κ(x) σ_θ(R) dW` | Neural Itô SDE. Storage temperature `x` enters only as a time change `s = κ(x) t` (Arrhenius `κ`, learned activation energy) |
| Spoilage `τ = inf{t : R_t ≥ b}` | First-passage time of the learned diffusion |
| Forecast `P(τ − t > h)` | Backward Kolmogorov PDE, Crank–Nicolson with upwinding; one table serves every temperature and horizon |
| Filtering between visits | Forward (Fokker–Planck) equation on a grid: an exact continuous-state HMM |
| Censored likelihood | Markov property + Brownian-bridge no-crossing factors; exact under the model ([derivation](docs/background/theory.md)) |
| Multi-dimensional latents | Latent-SDE ELBO with the Girsanov KL |
| Photo-only students from photo + scanner teachers | Auxiliary modality learning (AML): CAML, SAMD and the lab's distillation methods (`src/ripen/train/aml.py`, `samd.py`) |

The PDE solver matches the inverse-Gaussian first-passage law to 1e-3; parameter recovery on synthetic data is tested.

## Results

Each study has a plain-language Word report in [`results/`](results) and a technical write-up with every table in
[`docs/experiments/`](docs/experiments). File names start with the date the study was first run.

| Date | Study | Headline |
|---|---|---|
| 2026-10-28 | [Avocado: latent SDE vs 12 baselines](docs/experiments/2026-10-28_avocado-latent-sde-vs-baselines/report.md) (public set, 478 fruit, 5 seeds) | At an unseen storage temperature the Arrhenius clock cuts MAE 1.5–2.4× vs ablated clocks; with a photo only every 3rd day the latent SDE has CRPS 0.97 vs 1.5–2.7 for every other model; 80 % intervals cover 94 % (SDE) vs 64 % (latent ODE). A small transformer is best in-distribution. |
| 2026-11-06 | [Mango: photo vs microwave](docs/experiments/2026-11-06_mango-photo-vs-microwave/report.md) (48 lab fruit) | Microwave scans clearly beat photos; no model beats a known-age reference. |
| 2026-11-10 | [Avocado: microwave scanner views](docs/experiments/2026-11-10_avocado-microwave-scanner-views/report.md) | The 0°, 120° and 240° rotation views beat 60°, 180° and 300° (14 of 16 paired tests). |
| 2026-11-12 | [Views × time-series models × AML](docs/experiments/2026-11-12_views-time-series-models-and-aml/report.md) | Latent ODE is the best model on both fruits. Teacher-student gains on avocados come from the training schedule, not the scanner. |
| 2026-11-13 | [Results brief: ODE, SDE, AML](docs/experiments/2026-11-13_results-brief-ode-sde-aml/README.md) | The scanner barely helps even the teacher (0.66 vs 0.71); every gain over photo-only comes from training rounds. |

## Quickstart

```bash
pip install -e ".[dev]"        # Python 3.10+; JAX CPU wheels by default
make test                      # CPU test suite, incl. parameter recovery on a known SDE
make smoke                     # every model: build, loss, gradient, predict on synthetic data
python scripts/run/train.py model=latent_sde data=synthetic train=smoke
```

With the public avocado data (`python scripts/data/download_avocado.py`, then `make features`), each experiment is
one Hydra command, e.g. `python scripts/run/train.py model=latent_sde data=avocado +experiment=logo splits.held_out=T10`.
The full pipeline runs as SLURM jobs: `bash run_umiacs.sh pipeline` (`--local` to run without SLURM).

## Repository layout

```
src/ripen/          the package
  data/               loaders (public avocado, lab avocado + mango, synthetic), item schema, splits
  models/             latent SDE / ODE, first-passage PDE and filter, temperature clocks, 12 baselines
  train/              training loop, AML / CAML distillation, SAMD
  eval/               metrics (censored log score, CRPS, coverage, C-index), recalibration, mango scoring
src/nsde/           general SDE solver with a gradient-method switch (backprop / checkpoint / adjoint)
configs/            Hydra configs: data, model, train, experiment; splits/ holds the fixed train/val/test splits
scripts/
  data/               download, ingest, feature extraction, split files
  run/                train, evaluate, smoke test, study drivers (mango CV, scanner views, AML)
  analysis/           summaries, recalibration, paired bootstrap, late fusion
  reports/            build the Word reports in results/ and their charts
tests/              pytest suite (CPU)
results/            the Word reports, one per study, dated
docs/
  experiments/        technical write-up + tables per study, dated
  background/         theory, literature review, dataset survey, proposal
  plans/ data/        research plans; notes on the lab microwave data
  progress_log.md     running log of what was done
foundations/        separate side project: neural-SDE lecture, gradient benchmark, local-vol demo
run_umiacs.sh       every experiment as SLURM jobs on the UMIACS cluster
outputs/            (git-ignored) checkpoints, predictions, logs, regenerated summaries
data/               (git-ignored) raw data; see data/README.md
```

## Data

- **Public:** Mendeley "Hass" avocado photos ([10.17632/3xd9n945v8.1](https://doi.org/10.17632/3xd9n945v8.1),
  CC BY 4.0): 478 fruit, 14,710 photos, 3 storage temperatures.
- **Lab (not public):** microwave-scanned avocados and mangoes. Not in this repo; collaborators copy them from the
  cluster.

See [`data/README.md`](data/README.md) for the expected layout and the ingest commands.

## Working on it

- Tests and lint before pushing: `make test && make lint` (ruff, line length 120). CI runs the fast suite.
- On the UMIACS cluster, compute goes through `sbatch` / `srun`, never the login node; see
  [`.claude/skills/umiacs/SKILL.md`](.claude/skills/umiacs/SKILL.md) and `run_umiacs.sh`.
- New study: run outputs go to `outputs/<study>/`; add `docs/experiments/<YYYY-MM-DD>_<what-it-tests>/` with a
  `report.md` and the tables it cites, and put its Word report in `results/` under the same name.

## SDE foundations (side project)

`src/nsde` and [`foundations/`](foundations) hold earlier, non-fruit work: a graduate lecture on neural SDEs
(UMD CMSC 838B), a benchmark of gradient methods through SDE solvers (checkpointing is exact at ~1 MB and within
1.5× of backprop; the stochastic adjoint is constant-memory but 64× slower), and a neural local-volatility model
calibrated to a Heston smile by differentiating a Monte Carlo pricer (0.26 vol-point RMSE on held-out strikes).

## Key references

Li et al., *Scalable Gradients for SDEs*, AISTATS 2020 · Kidger et al., *Efficient and Accurate Gradients for Neural
SDEs*, NeurIPS 2021 · Rubanova et al., *Latent ODEs*, NeurIPS 2019 · Rindt et al., *Survival regression with proper
scoring rules*, AISTATS 2022 · Shen et al., *Auxiliary Modality Learning with Generalized Curriculum Distillation*
(SAMD), ICML 2023 · Liu et al., *CAML: Collaborative Auxiliary Modality Learning for Multi-Agent Systems*. Full list:
[`docs/background/literature_review.md`](docs/background/literature_review.md).

## License

MIT, see [LICENSE](LICENSE).
