import numpy as np
import pytest

from biomass.constraints import A
from biomass.cv import CVConfig, early_stopping_split, run_cv, select_ridge_alphas


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


def test_grouped_alpha_search_is_not_fooled_by_group_effects():
    # Features identify the group and the target is a group-level offset that
    # does not carry over to new groups: leave-one-out rewards memorising it,
    # grouped inner CV does not.
    rng = np.random.default_rng(0)
    n_groups, per_group, d = 12, 10, 30
    groups = np.repeat(np.arange(n_groups), per_group)
    centres = rng.normal(size=(n_groups, d))
    z = centres[groups] + rng.normal(0, 0.1, size=(len(groups), d))
    y = (rng.normal(0, 5, n_groups)[groups] + rng.normal(0, 0.1, len(groups)))[:, None]
    alphas = np.logspace(-2, 4, 13).tolist()
    loo = select_ridge_alphas(z, y, None, alphas, seed=0)
    grouped = select_ridge_alphas(z, y, groups, alphas, seed=0)
    assert grouped[0] == max(alphas)
    assert loo[0] < grouped[0]


def test_early_stopping_holdout_keeps_groups_together():
    groups = np.repeat(np.arange(15), 6)
    tr, es = early_stopping_split(len(groups), groups, CVConfig(head="lgbm"))
    assert len(es) > 0
    assert not set(groups[tr]) & set(groups[es])


def test_lgbm_final_models_use_the_whole_training_part(problem):
    X, Y, folds = problem
    cfg = CVConfig(head="lgbm", pca_components=8, lgbm_params={"n_estimators": 20})
    res = run_cv(X, Y, folds, cfg, keep_models=True)
    for model in res.models:
        for head in model.heads:
            assert head.n_jobs == 1
            assert head.booster_.num_trees() == head.n_estimators <= 20
