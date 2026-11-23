"""With / without background figure for the 2026-11-22 report (report env). Top row: original photos; bottom row:
the fruit-only cut-outs made by scripts/data/a2sl_fruit_features.py (data/fruit_only). Writes
outputs/report_figures/fruit_mask_examples.png."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _paths import FIGS, REPO
from PIL import Image

D = REPO / "data"
COLS = [  # (caption, original, fruit only)
    ("Lab avocado 2, day 1", "a2sl/avocado_jun2023/jpg/day01/02_side.jpg", "fruit_only/a2sl/day01/02_side.jpg"),
    ("Lab avocado 2, day 5", "a2sl/avocado_jun2023/jpg/day05/02_side.jpg", "fruit_only/a2sl/day05/02_side.jpg"),
    ("Lab avocado 2, day 9", "a2sl/avocado_jun2023/jpg/day09/02_side.jpg", "fruit_only/a2sl/day09/02_side.jpg"),
    ("Public avocado, day 1", "avocado/images/T10_d01_002_a_1.jpg", "fruit_only/public/images/T10_d01_002_a_1.jpg"),
    ("Public avocado, day 19", "avocado/images/T10_d19_002_a_5.jpg", "fruit_only/public/images/T10_d19_002_a_5.jpg"),
]

fig, ax = plt.subplots(2, len(COLS), figsize=(10, 4.4))
for j, (cap, a, b) in enumerate(COLS):
    for i, p in enumerate((a, b)):
        im = Image.open(D / p).convert("RGB")
        im.thumbnail((600, 600))
        ax[i, j].imshow(im)
        ax[i, j].set_xticks([]), ax[i, j].set_yticks([])
        for s in ax[i, j].spines.values():
            s.set_visible(False)
    ax[0, j].set_title(cap, fontsize=9)
ax[0, 0].set_ylabel("Original", fontsize=10)
ax[1, 0].set_ylabel("Background removed", fontsize=10)
fig.tight_layout()
out = FIGS / "fruit_mask_examples.png"
fig.savefig(out, dpi=200)
print("wrote", out)
