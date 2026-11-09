"""Combining OOF predictions of several runs.

One non-negative weight per model, shared by all five targets. A non-negative
combination of predictions that each satisfy the additive identities (and are
non-negative) satisfies them too, so no re-projection is needed afterwards.

``best`` and ``crossfit_nnls`` choose the weights on the other folds and apply
them to the held-out fold, so their OOF scores are not inflated by choosing on
the rows they are evaluated on. The weights returned for the test set are
fitted on all rows.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from biomass.data import weight_vector
from biomass.metric import weighted_r2

METHODS = ("mean", "best", "crossfit_nnls")


def _stack(preds: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(p, dtype=float) for p in preds], axis=-1)  # (n, 5, m)


def fit_nnls_weights(preds: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Weights ``w >= 0`` minimising the metric-weighted squared error of ``sum_m w_m P_m``."""
    stacked = _stack(preds)
    sqrt_w = np.sqrt(weight_vector())[None, :, None]
    design = (stacked * sqrt_w).reshape(-1, stacked.shape[-1])
    target = (np.asarray(y, dtype=float) * sqrt_w[..., 0]).ravel()
    weights, _ = nnls(design, target)
    return weights


def combine(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return _stack(preds) @ np.asarray(weights, dtype=float)


def pick_best(preds: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """One-hot weights for the run with the highest metric."""
    return np.eye(len(preds))[int(np.argmax([weighted_r2(y, p) for p in preds]))]


def _crossfit(preds: list[np.ndarray], y: np.ndarray, folds: np.ndarray, fit) -> tuple:
    folds = np.asarray(folds)
    oof = np.empty_like(np.asarray(preds[0], dtype=float))
    fold_weights = []
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        w = fit([p[tr] for p in preds], y[tr])
        oof[va] = combine([p[va] for p in preds], w)
        fold_weights.append(w)
    return oof, np.array(fold_weights)


def ensemble_oof(preds: list[np.ndarray], y: np.ndarray, folds: np.ndarray, method: str) -> dict:
    """Return the OOF of the ensemble, its score and the weights to use on test."""
    if method == "mean":
        weights = np.full(len(preds), 1.0 / len(preds))
        oof = combine(preds, weights)
        return {"oof": oof, "score": weighted_r2(y, oof), "weights": weights}

    fit = {"best": pick_best, "crossfit_nnls": fit_nnls_weights}.get(method)
    if fit is None:
        raise ValueError(f"unknown ensemble method: {method!r}")
    oof, fold_weights = _crossfit(preds, y, folds, fit)
    return {
        "oof": oof,
        "score": weighted_r2(y, oof),
        "weights": fit(preds, y),
        "fold_weights_mean": fold_weights.mean(axis=0),
        "fold_weights_std": fold_weights.std(axis=0),
    }
