"""How far the half-synthetic photos (ripen.data.a2sl_synth) are from the tracked fruit's real day-7 / 9 photos."""

import numpy as np
from omegaconf import OmegaConf

from ripen.pipeline import load_items

base = OmegaConf.load("configs/data/a2sl.yaml")
base.scan_standardize = True
real = {it.item_id: it for it in load_items(base)}
for m in ("linear", "logistic"):
    cfg = OmegaConf.merge(base, {"synthetic": m})
    items = load_items(cfg)
    print(m, len(items), "fruit:", [it.item_id for it in items])
    print("  visits", {it.n_visits for it in items}, "spoil", {(it.spoil_lo, it.spoil_hi) for it in items})
    for f in ("avo01", "avo02", "avo03", "avo11"):
        s, r = {it.item_id: it for it in items}[f], real[f]
        d = [float(np.linalg.norm(s.visits[i].features[0, :512] - r.visits[i].features[0, :512])) for i in (3, 4)]
        dr = float(np.linalg.norm(r.visits[3].features[0, :512] - r.visits[2].features[0, :512]))
        print(f"  {f}: |synthetic - real| photo day 7, 9 = {d[0]:.1f}, {d[1]:.1f}  (real day5->7 change {dr:.1f})")
    print("  finite:", all(np.isfinite(v.features).all() for it in items for v in it.visits))
