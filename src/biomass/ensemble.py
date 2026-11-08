"""Combining OOF predictions of several runs.

One non-negative weight per model, shared by all five targets. A non-negative
combination of predictions that each satisfy the additive identities (and are
non-negative) satisfies them too, so no re-projection is needed afterwards.

``crossfit_nnls`` learns the weights on the other folds and applies them to the
held-out fold, so its OOF score is not inflated by fitting the weights on the
rows it is evaluated on.
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


def ensemble_oof(preds: list[np.ndarray], y: np.ndarray, folds: np.ndarray, method: str) -> dict:
    """Return the OOF of the ensemble, its score and the weights to use on test."""
    m = len(preds)
    if method == "mean":
        weights = np.full(m, 1.0 / m)
        oof = combine(preds, weights)
        return {"oof": oof, "score": weighted_r2(y, oof), "weights": weights}

    if method == "best":
        scores = [weighted_r2(y, p) for p in preds]
        weights = np.eye(m)[int(np.argmax(scores))]
        oof = combine(preds, weights)
        return {"oof": oof, "score": weighted_r2(y, oof), "weights": weights}

    if method == "crossfit_nnls":
        folds = np.asarray(folds)
        oof = np.empty_like(np.asarray(preds[0], dtype=float))
        fold_weights = []
        for k in np.unique(folds):
            tr, va = folds != k, folds == k
            w = fit_nnls_weights([p[tr] for p in preds], y[tr])
            oof[va] = combine([p[va] for p in preds], w)
            fold_weights.append(w)
        fold_weights = np.array(fold_weights)
        return {
            "oof": oof,
            "score": weighted_r2(y, oof),
            "weights": fit_nnls_weights(preds, y),
            "fold_weights_mean": fold_weights.mean(axis=0),
            "fold_weights_std": fold_weights.std(axis=0),
        }

    raise ValueError(f"unknown ensemble method: {method!r}")
