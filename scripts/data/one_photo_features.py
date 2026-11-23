"""Photo features for the one-photo study (docs/plans/2026-11-22_one-photo-app.md), the same kind for the public
avocados (training) and the lab avocados (test):

* ``avocado_fruit_resnet18.npz``: ResNet-18 of the public fruit-only cut-outs (data/fruit_only/public, made by
  a2sl_fruit_features.py; the lab counterpart is ``a2sl_fruit_resnet18.npz``);
* ``avocado_colour_pub.npz`` / ``a2sl_colour_pub.npz``: the 12 skin-colour numbers, both z-scored with the public
  photos' mean and spread (the lab ``a2sl_colour.npz`` was scaled on lab photos, so it is not comparable).

python scripts/data/one_photo_features.py [--device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ripen.data.a2sl import load_a2sl_avocado, photo_key
from ripen.data.avocado import load_avocado
from ripen.data.features import extract_features, save_feature_cache
from ripen.data.fruit_mask import colour_features, fruit_mask

OUT = Path("data/features")
CUT = Path("data/fruit_only/public/images")


def colours(paths: list[str], keep: int) -> np.ndarray:
    out = []
    for i, p in enumerate(paths):
        im = Image.open(p)
        im.thumbnail((keep, keep))  # the sizes a2sl_fruit_features.py cut at
        out.append(colour_features(im, fruit_mask(im)))
        if i % 1000 == 0:
            print(f"  colour {i}/{len(paths)}", flush=True)
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    pub = [p for it in load_avocado("data/avocado") for v in it.visits for p in v.image_paths if Path(p).exists()]
    lab = [p for it in load_a2sl_avocado("data/manifests") for v in it.visits for p in v.image_paths]
    print(f"{len(pub)} public photos, {len(lab)} lab photos", flush=True)
    cut = [str(CUT / Path(p).name) for p in pub]
    missing = [c for c in cut if not Path(c).exists()]
    assert not missing, f"{len(missing)} public cut-outs missing, e.g. {missing[:3]}"
    f = extract_features(cut, "resnet18", 64, a.device, key_fn=lambda p: Path(p).name)
    save_feature_cache(OUT / "avocado_fruit_resnet18.npz", f)
    cp, cl = colours(pub, 512), colours(lab, 1024)
    mu, sd = cp.mean(0), cp.std(0) + 1e-6
    save_feature_cache(OUT / "avocado_colour_pub.npz",
                       {Path(p).name: ((cp[i] - mu) / sd).astype(np.float32) for i, p in enumerate(pub)})  # fmt: skip
    save_feature_cache(OUT / "a2sl_colour_pub.npz",
                       {photo_key(p): ((cl[i] - mu) / sd).astype(np.float32) for i, p in enumerate(lab)})  # fmt: skip
    print("lab colour, mean z by day folder (public scale):")
    for d in sorted({Path(p).parent.name for p in lab}):
        sel = [i for i, p in enumerate(lab) if Path(p).parent.name == d]
        print(f"  {d}: " + " ".join(f"{x:+.2f}" for x in ((cl[sel] - mu) / sd).mean(0)[:4]))


if __name__ == "__main__":
    main()
