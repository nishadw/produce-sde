# Neural SDE First-Passage Models

[![tests](https://github.com/nishadw/produce-sde/actions/workflows/tests.yml/badge.svg)](https://github.com/nishadw/produce-sde/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-1F2328)
![jax](https://img.shields.io/badge/JAX-Diffrax%20%7C%20Equinox-C8102E)
[![license](https://img.shields.io/badge/license-MIT-6B7280)](LICENSE)

**Learn a latent Itô diffusion from irregular, censored observations, and predict a calibrated distribution
over *when* it first hits a boundary. The likelihood is exact: it comes from the Kolmogorov PDEs, not from
path simulation.**

The application is avocado spoilage from daily photos (and, on a small set, microwave scans). Built in JAX /
Equinox / Diffrax.

## Methods

| Piece | Stochastic-calculus tool |
|---|---|
| Latent dynamics `dR = κ(x) μ_θ(R) dt + √κ(x) σ_θ(R) dW` | Neural Itô SDE; the condition `x` (temperature) enters only as a **time change** `s = κ(x) t` (Arrhenius `κ`, learned activation energy) |
| Event `τ = inf{t : R_t ≥ b}` | First-passage time of a learned diffusion |
| Forecast `P(τ − t > h)` | **Backward Kolmogorov PDE** `∂_s S = μ ∂_r S + ½σ² ∂_rr S`, Crank–Nicolson with upwinding; one table serves every condition and horizon |
| Filtering between visits | **Forward (Fokker–Planck) equation** on a grid: an exact continuous-state HMM (`fpt_filter`) |
| Interval-censored likelihood | Markov property + **Brownian-bridge** no-crossing factors `1 − exp(−2(b−r_k)(b−r_{k+1})/(σ²κΔ_k))`; exact under the model ([proof](docs/theory.md)) |
| Multi-dimensional latents | Latent-SDE ELBO with the **Girsanov** KL `½∫‖(f_post − f_prior)/σ‖² dt` (`latent_sde`) |
| Gradients through the solver | Backprop, recursive checkpointing, or the **stochastic adjoint** (one config switch) |

The PDE solver matches the inverse-Gaussian first-passage law to 1e-3. The derivations (exactness, properness
of the censored log score, identifiability of the clock) are in [`docs/theory.md`](docs/theory.md).

## Results

Avocado numbers are mean ± std over **5 independent training seeds** on the UMIACS GPU cluster (fixed split). The
full write-up with figures and every table is [`report/REPORT.md`](report/REPORT.md).

| | Result |
|---|---|
| **Transfer to an unseen storage temperature** | The Arrhenius clock beats both ablated clocks on every held-out group, cutting MAE **1.5–2.4×** (guided model: 1.1 / 1.4 / 5.5 days on T20 / ambient / T10). On cold storage (T10) it beats every learned baseline (transformer 7.9, GRU-D 7.1), but the classical Wiener degradation process does better still (3.0 days, 74 % coverage) |
| **Sparse observations** | With a photo every 3rd day the latent SDE has CRPS 0.97 vs 1.5–2.7 for every other model, including the transformer (1.89); significant against the transformer, Wiener and GRU-D at both sparse rates (paired bootstrap) |
| **In-distribution** | Guided + recalibrated latent SDE: CRPS 0.85 ± 0.01, MAE 1.20 days, 93 % coverage. Statistically tied with GRU + gap, ODE-RNN and GRU-D; a small transformer is significantly better (0.74 ± 0.06); neural CDE, Wiener, gamma and landmark Cox are significantly worse |
| **SDE vs ODE calibration** | Same ranking (C ≈ 0.93), but 80% intervals cover **94 ± 2%** for the SDE vs **64 ± 4%** for the deterministic latent ODE |
| **vs the published baseline** | Every distributional model beats classify-then-regress (CRPS 1.81) |

The SDE result is about robustness and transfer, not raw in-distribution accuracy, where a transformer is best.

On synthetic data the latent SDE's clock rates used to come out 14–24 % too fast; a guided posterior fixes that
(now +2–10 %, σ 0.256 vs 0.25). With guidance, the exact PDE likelihood also catches up with Monte Carlo on avocado
(CRPS 0.85 vs 0.87 recalibrated).

Baselines: 7 original (stage → linear, direct regressor, discrete survival, GRU + gap, GRU-D, ODE-RNN, latent ODE)
plus 5 added in Round 2 (Wiener and gamma degradation processes with an Arrhenius clock, landmark Cox, neural CDE,
transformer). See `report/REPORT.md`, "Round 2b", "New baselines" and "Paired significance tests".

Still open: cold-storage intervals are too narrow for every learned model.

**Mango study** (`report/MANGO_REPORT.md`, `bash run_umiacs.sh mango`): 48 mangoes cut open on days 0–14, microwave
scans every other day, photos on the cut day. Every model × {photo, microwave, both, photo-only distilled from both
with SAMD}, scored identically. Microwave clearly beats photo; SAMD does not make the photo-only models better than a
simple photo → age → batch-curve reference; nothing tells fruit of the same age apart.

## SDE foundations (`src/nsde`)

- **Gradient methods benchmark.** Backprop is exact but its memory grows with step count. Checkpointing is
  exact at ~1 MB and within 1.5× of backprop's speed. The stochastic adjoint uses constant memory but is 64×
  slower, and its error falls like 1/steps.
  ![Adjoint trade-offs](assets/adjoint_tradeoffs.png)
- **Neural local-volatility calibration.** Fits a neural local-vol SDE to a Heston implied-vol smile by
  differentiating a Monte Carlo pricer end to end. RMSE is 0.26 vol points on held-out strikes, vs 4.2 for the
  best flat Black–Scholes vol.
  ![Local-vol calibration](assets/local_vol_calibration.png)

## Run it

```bash
pip install -e ".[dev]"
make test                      # CPU tests, incl. parameter recovery on a known SDE
make smoke                     # every model: build, loss, gradient, predict
bash run_umiacs.sh pipeline    # full experiment pipeline on SLURM (--local to run without it)
```

Each experiment is one Hydra command, e.g.
`python scripts/train.py model=latent_sde data=avocado +experiment=logo splits.held_out=T10`.

## Layout

```
src/ripen/   data loaders, models (latent SDE/ODE, fpt_pde, clocks, 12 baselines), trainer, metrics
src/nsde/    SDE solver with gradient-method switch; Black–Scholes / Heston pricers
configs/     Hydra configs        scripts/  train, evaluate, summarize, figures
report/      results report (REPORT.md), the mango study (MANGO_REPORT.md), one-page Word summary (report/brief/)
docs/        theory, literature, data notes, plans
lecture/     graduate lecture on neural SDEs (slides, notes, demo)
```

## Data

The avocado photos come from Mendeley ([10.17632/3xd9n945v8.1](https://doi.org/10.17632/3xd9n945v8.1), CC BY 4.0):
478 fruit, 14,710 photos. The second dataset is A2SL microwave scans + photos of avocados. Raw data is not
committed; see [`data/README.md`](data/README.md).

## Lecture

*Neural SDEs: Learning Random Dynamics.* A graduate lecture for UMD CMSC 838B (Fall 2026). It covers neural
ODEs, the adjoint method, Brownian motion and Itô's lemma, Euler–Maruyama, pathwise SDE training and latent
SDEs. Slides and demo are in [`lecture/`](lecture/).

## Key references

Li et al., *Scalable Gradients for SDEs*, AISTATS 2020 · Kidger et al., *Efficient and Accurate Gradients for
Neural SDEs*, NeurIPS 2021 · Rubanova et al., *Latent ODEs*, NeurIPS 2019 · Rindt et al., *Survival regression
with proper scoring rules*, AISTATS 2022. The full list is in [`docs/literature_review.md`](docs/literature_review.md).
