"""Embedding code paths. The transformer tests need the [embed] extra and build
randomly initialised models (pretrained=False), so nothing is downloaded."""

import numpy as np
import pytest
from PIL import Image

from biomass.embed import color_stats_encoder, extract, image_views


def _write_images(tmp_path, n=3, size=(96, 48)):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
        p = tmp_path / f"img{i}.jpg"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    return paths


def test_wide_image_gives_two_halves_and_flips():
    img = Image.new("RGB", (200, 100))
    assert len(image_views(img, 32, split_halves=True, flip=True)) == 4
    assert len(image_views(img, 32, split_halves=False, flip=True)) == 2
    assert len(image_views(img, 32, split_halves=True, flip=False)) == 2
    assert image_views(img, 32)[0].size == (32, 32)


def test_view_average_ignores_half_order():
    rng = np.random.default_rng(0)
    left = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
    right = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
    a = Image.fromarray(np.concatenate([left, right], axis=1))
    b = Image.fromarray(np.concatenate([right, left], axis=1))

    def feat(img):
        views = np.stack([np.asarray(v, np.float32) / 255 for v in image_views(img, 40)])
        return color_stats_encoder(views).mean(axis=0)

    np.testing.assert_allclose(feat(a), feat(b), rtol=1e-5)


def test_extract_with_color_stats(tmp_path):
    paths = _write_images(tmp_path)
    X = extract(paths, color_stats_encoder, image_size=32, batch_size=2)
    assert X.shape[0] == 3 and X.dtype == np.float32
    assert np.isfinite(X).all()


@pytest.mark.embed
def test_timm_dinov3_random_weights(tmp_path):
    pytest.importorskip("timm")
    from biomass.embed import TimmEncoder

    enc = TimmEncoder(
        "vit_base_patch16_dinov3.lvd1689m", image_size=64, pretrained=False, device="cpu"
    )
    X = extract(_write_images(tmp_path, n=2), enc, image_size=64, batch_size=2)
    assert X.shape == (2, 2 * enc.model.num_features)
    assert np.isfinite(X).all()


@pytest.mark.embed
def test_siglip_random_weights(tmp_path):
    pytest.importorskip("transformers")
    from transformers import SiglipVisionConfig

    from biomass.embed import SiglipEncoder

    config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=64,
        patch_size=16,
    )
    enc = SiglipEncoder("unused", pretrained=False, config=config, device="cpu")
    X = extract(_write_images(tmp_path, n=2), enc, image_size=enc.image_size, batch_size=2)
    assert X.shape == (2, 64)
    assert np.isfinite(X).all()
