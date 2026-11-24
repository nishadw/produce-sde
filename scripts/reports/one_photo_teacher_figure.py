"""Example forecasts for the phase-2 report (ripen env: reads predictions only). Lab avocado 2 on days 1, 3, 5, 7
(fruit cut-outs); below, the odds of spoiling on each day from that one photo for the phase-1 public model, the lab
fine-tune and the scan-taught students (CAML, SAMD), with the actual spoil window shaded. avo02 is held out in fold 1.

python scripts/reports/one_photo_teacher_figure.py --run transformer_colour --seed 1
Writes outputs/report_figures/one_photo_teacher_examples_{run}.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _paths import FIGS, REPO
from PIL import Image

FRUIT, FOLD, DAYS, SHOW = "avo02", 1, (1, 3, 5, 7), 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    s = f"seed{a.seed}"
    t = REPO / "outputs" / "one_photo_teacher" / a.run / s / f"fold{FOLD}"
    rows = (("Public\nonly", REPO / "outputs" / "one_photo" / a.run / s / "predictions_lab.npz"),
            ("Fine-tuned\non lab", t / "finetune" / "predictions_test.npz"),
            ("+ scan teacher\n(CAML)", t / "kd" / "predictions_test.npz"),
            ("+ scan teacher\n(SAMD)", t / "samd" / "predictions_test.npz"))  # fmt: skip
    fig, ax = plt.subplots(1 + len(rows), len(DAYS), figsize=(10, 7.6),
                           gridspec_kw={"height_ratios": [1.2] + [1] * len(rows)})  # fmt: skip
    for j, day in enumerate(DAYS):
        im = Image.open(REPO / "data" / "fruit_only" / "a2sl" / f"day{day:02d}" / "02_side.jpg")
        ax[0, j].imshow(im)
        ax[0, j].axis("off")
        ax[0, j].set_title(f"Photo on day {day}", fontsize=9)
    for r, (label, path) in enumerate(rows, start=1):
        z = np.load(path)
        ids = [str(i) for i in z["item_ids"]]
        for j in range(len(DAYS)):
            i = ids.index(f"{FRUIT}@{j}")
            F, lo, hi = z["F"][i], float(z["lo"][i]), float(z["hi"][i])
            pmf = np.diff(np.r_[0.0, F])[: SHOW + 1]
            med = int(np.argmax(F >= 0.5))
            b = ax[r, j]
            b.axvspan(lo + 0.5, hi + 0.5, color="#E3F0DC", zorder=0)
            b.bar(np.arange(SHOW + 1), pmf, color="#3B6EA8", width=0.8, zorder=2)
            b.set_xlim(-0.6, SHOW + 0.6)
            b.set_ylim(0, max(0.35, pmf.max() * 1.15))
            b.tick_params(labelsize=7)
            b.spines[["top", "right"]].set_visible(False)
            b.set_title(f"says ~{med} day{'' if med == 1 else 's'}; actually {int(lo) + 1}–{int(hi)}", fontsize=8)
            if j == 0:
                b.set_ylabel(label, fontsize=8.5, fontweight="bold")
            if r == len(rows):
                b.set_xlabel("days from photo", fontsize=8)
    fig.tight_layout()
    out = FIGS / f"one_photo_teacher_examples_{a.run}.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
