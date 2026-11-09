import pytest

from biomass.config import apply_overrides


def test_overrides_parse_yaml_values_and_create_sections():
    cfg = {"cv": {"scheme": "group_date", "seeds": [0]}, "backbone": "dinov3_large"}
    out = apply_overrides(
        cfg, ["cv.seeds=[1, 2]", "backbone=color_stats", "model.pca_components=null"]
    )
    assert out["cv"] == {"scheme": "group_date", "seeds": [1, 2]}
    assert out["backbone"] == "color_stats"
    assert out["model"] == {"pca_components": None}
    assert cfg["cv"]["seeds"] == [0]  # input left untouched


def test_override_without_equals_sign_is_rejected():
    with pytest.raises(ValueError):
        apply_overrides({}, ["cv.scheme"])
