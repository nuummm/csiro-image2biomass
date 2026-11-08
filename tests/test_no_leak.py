import numpy as np
import pytest

from biomass.constraints import A
from biomass.cv import CVConfig, run_cv


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    n, d = 90, 40
    X = rng.normal(size=(n, d)) + np.arange(n)[:, None] * 0.01
    comps = np.abs(X[:, :3] @ rng.normal(size=(3, 3))) * 5 + 1
    return X, comps @ A.T, np.arange(n) % 5


@pytest.mark.parametrize("head", ["ridge", "lgbm"])
def test_preprocessing_is_fitted_on_train_part_only(problem, head):
    X, Y, folds = problem
    cfg = CVConfig(head=head, pca_components=8, lgbm_params={"n_estimators": 20})
    res = run_cv(X, Y, folds, cfg, keep_models=True)
    assert len(res.models) == 5
    for k, model in enumerate(res.models):
        train = folds != k
        np.testing.assert_array_equal(model.train_idx, np.where(train)[0])
        scaler = model.preprocessor.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, X[train].mean(axis=0))
        np.testing.assert_allclose(scaler.scale_, X[train].std(axis=0))
        assert not np.allclose(scaler.mean_, X.mean(axis=0))


def test_oof_rows_come_from_models_that_never_saw_them(problem):
    X, Y, folds = problem
    cfg = CVConfig(pca_components=8)
    res = run_cv(X, Y, folds, cfg, keep_models=True)
    for k, model in enumerate(res.models):
        assert not np.isin(np.where(folds == k)[0], model.train_idx).any()
