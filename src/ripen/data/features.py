"""Image features from a frozen pretrained backbone, cached to disk once.

The JAX models never touch pixels: they read ``(D,)`` vectors per photo from
an ``.npz`` cache keyed by filename. Extraction needs the optional
``[features]`` extra (torch, torchvision, pillow); fine-tuning the backbone
is out of scope for the CPU path and is left as a documented option.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

BACKBONE_DIMS = {"resnet18": 512, "resnet50": 2048, "convnext_tiny": 768, "dinov2_vits14": 384, "dinov2_vitb14": 768}


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_feature_cache(path: str | Path, feats: dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **feats)


class HashFeaturizer:
    """Deterministic stand-in for a backbone: features are a hash of the filename. For tests and smoke runs."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def __call__(self, paths: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for p in paths:
            seed = int.from_bytes(hashlib.sha1(Path(p).name.encode()).digest()[:8], "little")
            out[Path(p).name] = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
        return out


def extract_features(
    paths: list[str], backbone: str = "resnet18", batch_size: int = 64, device: str = "cpu", image_size: int = 224
) -> dict[str, np.ndarray]:
    """Run a frozen ImageNet backbone over ``paths``; returns ``{filename: (D,) float32}``.

    Uses global-average-pooled features (the layer before the classifier).
    """
    import torch
    import torchvision
    from PIL import Image

    if backbone.startswith("dinov2"):
        return _extract_dinov2(paths, backbone, batch_size, device, image_size)
    weights = {
        "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
        "convnext_tiny": torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    }[backbone]
    model = getattr(torchvision.models, backbone)(weights=weights)
    if backbone.startswith("resnet"):
        model.fc = torch.nn.Identity()
    else:
        model.classifier[-1] = torch.nn.Identity()
    model.eval().to(device)
    tf = weights.transforms()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            f = model(x).float().cpu().numpy()
            for p, v in zip(chunk, f, strict=True):
                out[Path(p).name] = v.astype(np.float32)
    return out


def _extract_dinov2(paths: list[str], backbone: str, batch_size: int, device: str, image_size: int):
    """Self-supervised DINOv2 ViT (Oquab et al. 2024) via ``torch.hub``; the feature is the CLS token.

    DINOv2 features keep colour and texture that ImageNet-supervised pooling partly discards, which is what ripeness
    is. Weights are fetched once into ``$TORCH_HOME/hub``.
    """
    import torch
    from PIL import Image
    from torchvision import transforms as T

    model = torch.hub.load("facebookresearch/dinov2", backbone).eval().to(device)
    tf = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            f = model(x).float().cpu().numpy()
            for p, v in zip(chunk, f, strict=True):
                out[Path(p).name] = v.astype(np.float32)
    return out


def standardize_cache(feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Z-score every dimension over all photos (unsupervised; uses no labels). ViT features have a few
    very large dimensions that would otherwise dominate the visit encoder's first layer."""
    X = np.stack(list(feats.values()))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return {k: ((v - mu) / sd).astype(np.float32) for k, v in feats.items()}
