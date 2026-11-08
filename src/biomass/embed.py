"""Frozen image embeddings.

Images are about twice as wide as they are tall, so each one is cut into a left
and a right square half; each half is optionally mirrored. Every view goes
through a frozen encoder and the view features are averaged into one vector per
image (the average does not depend on which half is "left").

For transformer backbones the feature of a view is ``[CLS (or pooled) token,
mean of patch tokens]``. torch / timm / transformers are imported lazily so the
rest of the package works without the ``[embed]`` extra. ``color_stats`` is a
numpy-only baseline used by the tests and as a sanity floor.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

Encoder = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BackboneSpec:
    kind: str  # "timm" | "hf_siglip" | "color"
    model_name: str
    image_size: int


BACKBONES: dict[str, BackboneSpec] = {
    "dinov3_large": BackboneSpec("timm", "vit_large_patch16_dinov3.lvd1689m", 448),
    "dinov3_base": BackboneSpec("timm", "vit_base_patch16_dinov3.lvd1689m", 448),
    "siglip_so400m": BackboneSpec("hf_siglip", "google/siglip-so400m-patch14-384", 384),
    "color_stats": BackboneSpec("color", "", 256),
}


def select_device(preferred: str | None = None) -> str:
    """``cuda`` > ``mps`` > ``cpu``; ``$BIOMASS_DEVICE`` or ``preferred`` overrides."""
    choice = preferred or os.environ.get("BIOMASS_DEVICE")
    if choice:
        return choice
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- views


def image_views(img: Image.Image, size: int, split_halves: bool = True, flip: bool = True):
    """Return the list of ``size x size`` RGB views of one image."""
    img = img.convert("RGB")
    w, h = img.size
    if split_halves and w >= 1.5 * h:
        parts = [img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))]
    else:
        parts = [img]
    views = []
    for part in parts:
        part = part.resize((size, size), Image.Resampling.BICUBIC)
        views.append(part)
        if flip:
            views.append(part.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    return views


def _batched(items: Sequence, batch_size: int) -> Iterator[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract(
    paths: Sequence[str],
    encoder: Encoder,
    image_size: int,
    batch_size: int = 16,
    split_halves: bool = True,
    flip: bool = True,
    progress: bool = False,
) -> np.ndarray:
    """Embed every image in ``paths`` and return an ``(n_images, dim)`` float32 array."""
    feats = []
    for k, chunk in enumerate(_batched(list(paths), batch_size)):
        views, owner = [], []
        for i, path in enumerate(chunk):
            with Image.open(path) as img:
                vs = image_views(img, image_size, split_halves, flip)
            views.extend(vs)
            owner.extend([i] * len(vs))
        batch = np.stack([np.asarray(v, dtype=np.float32) / 255.0 for v in views])
        out = encoder(batch)
        owner_arr = np.asarray(owner)
        feats.extend(out[owner_arr == i].mean(axis=0) for i in range(len(chunk)))
        done = min((k + 1) * batch_size, len(paths))
        if progress and (k % 10 == 9 or done == len(paths)):
            print(f"  {done}/{len(paths)}", flush=True)
    return np.stack(feats).astype(np.float32)


# --------------------------------------------------------------------------- encoders


def color_stats_encoder(batch: np.ndarray) -> np.ndarray:
    """Hand-crafted colour features: channel moments, excess-green statistics, green fraction."""
    r, g, b = batch[..., 0], batch[..., 1], batch[..., 2]
    exg = 2 * g - r - b
    flat = batch.reshape(len(batch), -1, 3)
    exg_flat = exg.reshape(len(batch), -1)
    feats = [
        flat.mean(axis=1),
        flat.std(axis=1),
        exg_flat.mean(axis=1, keepdims=True),
        exg_flat.std(axis=1, keepdims=True),
        np.quantile(exg_flat, [0.1, 0.5, 0.9], axis=1).T,
        (exg_flat > 0.05).mean(axis=1, keepdims=True),
    ]
    return np.concatenate(feats, axis=1).astype(np.float32)


class TimmEncoder:
    """A frozen timm ViT; returns ``[CLS, mean(patch tokens)]`` per view."""

    def __init__(
        self,
        model_name: str,
        image_size: int,
        pretrained: bool = True,
        device: str | None = None,
        weights: str | None = None,
    ) -> None:
        import timm
        import torch

        self.torch = torch
        self.device = select_device(device)
        # ``weights`` points timm at a local checkpoint (offline environments).
        overlay = {"file": weights} if weights else None
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            pretrained_cfg_overlay=overlay,
            num_classes=0,
            img_size=image_size,
        )
        self.model = model.eval().to(self.device)
        cfg = model.pretrained_cfg
        self.mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
        self.n_prefix = int(getattr(model, "num_prefix_tokens", 1))

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(batch).permute(0, 3, 1, 2)
        x = ((x - self.mean) / self.std).to(self.device)
        with torch.inference_mode(), _autocast(torch, self.device):
            tokens = self.model.forward_features(x)
        tokens = tokens.float()
        cls = tokens[:, 0]
        patches = tokens[:, self.n_prefix :].mean(dim=1)
        return torch.cat([cls, patches], dim=1).cpu().numpy()


class SiglipEncoder:
    """A frozen SigLIP vision tower; returns ``[pooled output, mean(patch tokens)]``."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        config=None,
        device: str | None = None,
        weights: str | None = None,
    ) -> None:
        import torch
        from transformers import SiglipVisionModel

        self.torch = torch
        self.device = select_device(device)
        if pretrained:
            # ``weights`` is a local directory saved with ``save_pretrained`` (offline use).
            model = SiglipVisionModel.from_pretrained(weights or model_name)
        else:
            if config is None:
                raise ValueError("pretrained=False needs an explicit SiglipVisionConfig")
            model = SiglipVisionModel(config)
        self.model = model.eval().to(self.device)
        self.image_size = int(model.config.image_size)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(batch).permute(0, 3, 1, 2)
        x = ((x - 0.5) / 0.5).to(self.device)
        with torch.inference_mode(), _autocast(torch, self.device):
            out = self.model(pixel_values=x)
        pooled = out.pooler_output.float()
        patches = out.last_hidden_state.float().mean(dim=1)
        return torch.cat([pooled, patches], dim=1).cpu().numpy()


def _autocast(torch, device: str):
    # fp16 only on CUDA; MPS / CPU stay in fp32 for numerical safety.
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def build_encoder(
    backbone: str,
    pretrained: bool = True,
    device: str | None = None,
    weights: str | None = None,
) -> Encoder:
    spec = BACKBONES[backbone]
    if spec.kind == "color":
        return color_stats_encoder
    if spec.kind == "timm":
        return TimmEncoder(spec.model_name, spec.image_size, pretrained, device, weights)
    if spec.kind == "hf_siglip":
        return SiglipEncoder(spec.model_name, pretrained, device=device, weights=weights)
    raise ValueError(f"unknown backbone kind: {spec.kind!r}")
