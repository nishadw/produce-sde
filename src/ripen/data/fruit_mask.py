"""Cut the fruit out of a photo taken on a plain background (A2SL avocados: beige board).

The background colour is the median of the image border; a pixel is fruit when its colour is far from it (Otsu
threshold on the distance, so no fixed cut-off), keeping the largest connected blob with its holes filled. Shadows are
darker but keep the background's hue, so the distance uses chromaticity as well as brightness and a shadow falls under
the threshold. ``fruit_only`` sets everything else to a neutral grey; ``colour_features`` summarises the fruit's
pixels.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def _otsu(x: np.ndarray, bins: int = 256) -> float:
    h, e = np.histogram(x, bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def fruit_mask(img: Image.Image, work: int = 256) -> np.ndarray:
    """Boolean mask at the image's own size."""
    small = np.asarray(img.convert("RGB").resize((work, int(work * img.height / img.width))), np.float32) / 255.0
    border = np.concatenate([small[:8].reshape(-1, 3), small[-8:].reshape(-1, 3), small[:, :8].reshape(-1, 3),
                             small[:, -8:].reshape(-1, 3)])  # fmt: skip
    bg = np.median(border, 0)
    s = small.sum(-1, keepdims=True) + 1e-3
    chrom = small / s - bg / bg.sum()  # hue-like: unchanged by shadow
    bright = np.log(small.mean(-1) + 1e-3) - np.log(bg.mean() + 1e-3)
    d = np.sqrt(np.sum(chrom**2, -1)) * 6.0 + np.maximum(-bright, 0) * 0.35
    m = d > _otsu(d)
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.ones((img.height, img.width), bool)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = ndimage.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((img.width, img.height))) > 127


def fruit_only(img: Image.Image, mask: np.ndarray | None = None, crop: bool = True) -> Image.Image:
    """The fruit on neutral grey, cropped to its bounding box (plus a 5 % margin) so it fills the frame."""
    rgb = np.asarray(img.convert("RGB")).copy()
    m = fruit_mask(img) if mask is None else mask
    rgb[~m] = 128
    out = Image.fromarray(rgb)
    if crop and m.any():
        ys, xs = np.where(m)
        pad = int(0.05 * max(np.ptp(ys), np.ptp(xs)))
        out = out.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, img.width),
                        min(ys.max() + pad, img.height)))  # fmt: skip
    return out


def colour_features(img: Image.Image, mask: np.ndarray | None = None) -> np.ndarray:
    """12 numbers about the fruit's skin: mean / spread / 10th / 90th percentile of brightness, green-red and
    blue-yellow (an approximate CIE Lab), over fruit pixels only."""
    m = fruit_mask(img) if mask is None else mask
    rgb = np.asarray(img.convert("RGB"), np.float32)[m] / 255.0
    L = rgb.mean(-1)
    a = rgb[:, 0] - rgb[:, 1]  # red minus green: rises as the skin turns from green to dark purple
    b = (rgb[:, 0] + rgb[:, 1]) / 2 - rgb[:, 2]
    out = []
    for ch in (L, a, b):
        out += [ch.mean(), ch.std(), np.percentile(ch, 10), np.percentile(ch, 90)]
    return np.asarray(out, np.float32)
