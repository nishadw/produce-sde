"""Example forecasts for the one-photo report (ripen env: reads predictions only, no JAX). Top: lab avocado 2 on
days 1, 3, 5, 7 (fruit-only cut-outs); bottom: the chosen run's odds of spoiling on each day from the photo, with the
actual spoil window shaded. Writes outputs/report_figures/one_photo_examples.png.

python scripts/reports/one_photo_figure.py --run latent_sde_fruit [--seed 1]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _paths import FIGS, REPO
from PIL import Image

FRUIT, DAYS, SHOW = "avo02", (1, 3, 5, 7), 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    z = np.load(REPO / "outputs" / "one_photo" / a.run / f"seed{a.seed}" / "predictions_lab.npz")
    ids = [str(i) for i in z["item_ids"]]
    fig, ax = plt.subplots(2, len(DAYS), figsize=(10, 4.6), gridspec_kw={"height_ratios": [1.15, 1]})
    for j, day in enumerate(DAYS):
        i = ids.index(f"{FRUIT}@{j}")
        F, lo, hi = z["F"][i], float(z["lo"][i]), float(z["hi"][i])
        pmf = np.diff(np.r_[0.0, F])[: SHOW + 1]
        med = int(np.argmax(F >= 0.5))
        im = Image.open(REPO / "data" / "fruit_only" / "a2sl" / f"day{day:02d}" / "02_side.jpg")
        ax[0, j].imshow(im)
        ax[0, j].axis("off")
        ax[0, j].set_title(f"Photo on day {day}", fontsize=9)
        b = ax[1, j]
        b.axvspan(lo + 0.5, hi + 0.5, color="#E3F0DC", zorder=0)
        b.bar(np.arange(SHOW + 1), pmf, color="#3B6EA8", width=0.8, zorder=2)
        b.set_xlim(-0.6, SHOW + 0.6)
        b.set_ylim(0, max(0.35, pmf.max() * 1.15))
        b.set_xlabel("days from photo", fontsize=8)
        b.tick_params(labelsize=7)
        b.spines[["top", "right"]].set_visible(False)
        b.set_title(f"says ~{med} days; actually {int(lo) + 1}–{int(hi)}", fontsize=8.5)
        if j == 0:
            b.set_ylabel("odds of spoiling that day", fontsize=8)
    fig.tight_layout()
    out = FIGS / "one_photo_examples.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
