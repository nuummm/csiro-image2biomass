import numpy as np
import pytest

from biomass.constraints import (
    FITTED_TARGETS,
    A,
    compose,
    constraint_residuals,
    project_nnls,
)
from biomass.cv import CVConfig, predict_raw, to_model_space
from biomass.data import weight_vector

ENFORCING = ("sum", "diff", "nnls")


@pytest.fixture
def noisy():
    rng = np.random.default_rng(0)
    return rng.normal(20, 15, size=(200, 5))  # includes negatives on purpose


@pytest.mark.parametrize("formulation", ENFORCING)
def test_additive_identities_hold(formulation, noisy):
    fitted = noisy[:, : len(FITTED_TARGETS[formulation])]
    out = compose(formulation, fitted)
    assert out.shape == (len(noisy), 5)
    assert np.abs(constraint_residuals(out)).max() < 1e-6
    assert (out >= -1e-9).all()


def test_independent_only_clips(noisy):
    out = compose("independent", noisy)
    np.testing.assert_allclose(out, np.clip(noisy, 0, None))


def test_nnls_keeps_feasible_input_unchanged():
    rng = np.random.default_rng(1)
    comps = rng.gamma(2.0, 10.0, size=(50, 3))
    comps[::5, 2] = 0.0  # zero clover on the boundary of the feasible set
    y = comps @ A.T
    for weights in ("metric", "identity"):
        np.testing.assert_allclose(project_nnls(y, weights), y, atol=1e-8)


def test_nnls_components_are_non_negative(noisy):
    out = project_nnls(noisy)
    assert (out[:, :3] >= -1e-12).all()


def test_nnls_is_closer_than_sum_under_metric_weights(noisy):
    from biomass.data import weight_vector

    w = weight_vector()
    proj = project_nnls(noisy, "metric")
    naive = compose("sum", noisy[:, :3])
    d_proj = ((proj - noisy) ** 2 * w).sum(axis=1)
    d_naive = ((naive - noisy) ** 2 * w).sum(axis=1)
    assert (d_proj <= d_naive + 1e-9).all()


def test_diff_derives_children_by_subtraction():
    fitted = np.array([[100.0, 60.0, 45.0]])  # Total, GDM, Green
    out = compose("diff", fitted)
    np.testing.assert_allclose(out, [[45.0, 40.0, 15.0, 60.0, 100.0]])


def test_diff_rebuilds_parents_when_a_difference_is_negative():
    fitted = np.array([[50.0, 30.0, 40.0]])  # GDM < Green and Total < Green + Clover
    out = compose("diff", fitted)
    # Clover -> 0, GDM rebuilt as 40; Dead = 50 - 40 = 10.
    np.testing.assert_allclose(out, [[40.0, 10.0, 0.0, 40.0, 50.0]])
    fitted = np.array([[30.0, 50.0, 40.0]])  # Total < GDM
    np.testing.assert_allclose(compose("diff", fitted), [[40.0, 0.0, 10.0, 50.0, 50.0]])


@pytest.mark.parametrize("formulation", ENFORCING)
def test_constraints_hold_on_raw_scale_after_expm1(formulation):
    rng = np.random.default_rng(3)
    comps = rng.gamma(2.0, 10.0, size=(30, 3))
    y_raw = comps @ A.T
    cfg = CVConfig(target_space="log1p", formulation=formulation)
    idx = {"sum": [0, 1, 2], "diff": [4, 3, 0], "nnls": [0, 1, 2, 3, 4]}[formulation]
    log_pred = to_model_space(y_raw[:, idx], "log1p") + rng.normal(0, 0.1, (30, len(idx)))
    out = predict_raw(log_pred, cfg)
    assert np.abs(constraint_residuals(out)).max() < 1e-6
    # The same composition applied in log space would not be additive after expm1.
    wrong = np.expm1(compose(formulation, log_pred))
    assert np.abs(constraint_residuals(wrong)).max() > 1e-3


def test_projection_never_moves_away_from_a_feasible_truth():
    # Projecting onto a convex set is non-expansive in the W-norm, so when the
    # truth satisfies the identities the projected prediction is never farther from it.
    rng = np.random.default_rng(5)
    y = rng.gamma(2.0, 10.0, size=(500, 3)) @ A.T
    y_hat = y + rng.normal(0, 15, size=y.shape)
    w = weight_vector()

    def err(p):
        return (w * (p - y) ** 2).sum(axis=1)

    assert (err(project_nnls(y_hat)) <= err(y_hat) + 1e-9).all()
