import numpy as np
import pytest

from biomass.data import weight_vector
from biomass.metric import r2_per_target, weighted_r2


@pytest.fixture
def y():
    rng = np.random.default_rng(0)
    return rng.gamma(2.0, 10.0, size=(50, 5))


def test_perfect_prediction_scores_one(y):
    assert weighted_r2(y, y) == pytest.approx(1.0)


def test_global_weighted_mean_scores_zero(y):
    w = np.broadcast_to(weight_vector(), y.shape)
    y_bar = np.sum(w * y) / np.sum(w)
    assert weighted_r2(y, np.full_like(y, y_bar)) == pytest.approx(0.0, abs=1e-12)


def test_matches_long_format_definition(y):
    rng = np.random.default_rng(1)
    pred = y + rng.normal(0, 5, y.shape)
    # Flatten to one row per (image, target) and apply the Evaluation-page formula.
    w = np.tile(weight_vector(), len(y))
    yt, yp = y.ravel(), pred.ravel()
    y_bar = np.sum(w * yt) / np.sum(w)
    expected = 1 - np.sum(w * (yt - yp) ** 2) / np.sum(w * (yt - y_bar) ** 2)
    assert weighted_r2(y, pred) == pytest.approx(expected)


def test_is_not_the_mean_of_per_target_r2(y):
    rng = np.random.default_rng(2)
    pred = y + rng.normal(0, 8, y.shape)
    per_target = np.array(list(r2_per_target(y, pred).values()))
    assert weighted_r2(y, pred) != pytest.approx(float(per_target @ weight_vector()))


def test_shape_mismatch_raises(y):
    with pytest.raises(ValueError):
        weighted_r2(y, y[:, :4])
