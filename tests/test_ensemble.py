import numpy as np
import pytest

from biomass.constraints import A, constraint_residuals
from biomass.ensemble import combine, ensemble_oof, fit_nnls_weights


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 10.0, size=(100, 3)) @ A.T
    good = np.clip(y + rng.normal(0, 2, y.shape), 0, None)
    bad = np.clip(y + rng.normal(0, 20, y.shape), 0, None)
    folds = np.arange(100) % 5
    return y, [good, bad], folds


def test_nnls_prefers_the_better_model(setup):
    y, preds, _ = setup
    w = fit_nnls_weights(preds, y)
    assert (w >= 0).all()
    assert w[0] > w[1]


@pytest.mark.parametrize("method", ["mean", "best", "crossfit_nnls"])
def test_methods_return_full_oof(setup, method):
    y, preds, folds = setup
    res = ensemble_oof(preds, y, folds, method)
    assert res["oof"].shape == y.shape
    assert np.isfinite(res["score"])


def test_best_picks_the_better_model(setup):
    y, preds, folds = setup
    np.testing.assert_array_equal(ensemble_oof(preds, y, folds, "best")["weights"], [1, 0])


def test_non_negative_combination_keeps_identities():
    rng = np.random.default_rng(1)
    preds = [rng.gamma(2.0, 10.0, size=(20, 3)) @ A.T for _ in range(3)]
    out = combine(preds, np.array([0.2, 0.5, 0.7]))
    assert np.abs(constraint_residuals(out)).max() < 1e-9
