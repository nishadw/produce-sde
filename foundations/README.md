# foundations/

Earlier neural-SDE work, separate from the produce study. It uses the general solver in `src/nsde`.

| Folder | What it is |
|---|---|
| `lecture/` | *Neural SDEs: Learning Random Dynamics*, a graduate lecture for UMD CMSC 838B: slides, demo and build scripts |
| `experiments/adjoint_tradeoffs.py`, `adjoint_memory_scaling.py` | Memory, speed and gradient error of backprop vs checkpointing vs the stochastic adjoint |
| `experiments/local_vol_calibration.py` | Neural local-volatility SDE calibrated to a Heston smile through a differentiable Monte Carlo pricer |
| `experiments/demo_video/` | Renders the demo animation in `assets/` |
| `results/` | Numbers written by the experiments (JSON) |
| `assets/` | Figures and the demo animation used in the top-level README and the lecture |

Run from the repo root, e.g. `python foundations/experiments/adjoint_tradeoffs.py`.
