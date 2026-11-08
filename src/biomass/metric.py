"""Competition metric: a single weighted R^2 over all (image, target) rows.

From the Evaluation page: every row of the long-format file gets the weight of
its target (Total 0.5, GDM 0.2, the three components 0.1 each) and one R^2 is
computed over all rows together, with the *global* weighted mean of the ground
truth as the baseline. It is not an average of per-target R^2 values.
Inputs are always on the raw (gram) scale.
"""

from __future__ import annotations

import numpy as np

from biomass.data import TARGETS, weight_vector


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Official metric for ``(n_images, 5)`` matrices in ``TARGETS`` column order."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.shape[-1] != len(TARGETS):
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    w = weight_vector() if weights is None else np.asarray(weights, dtype=float)
    w_rows = np.broadcast_to(w, y_true.shape)

    y_bar = np.sum(w_rows * y_true) / np.sum(w_rows)
    ss_res = np.sum(w_rows * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w_rows * (y_true - y_bar) ** 2)
    return float(1.0 - ss_res / ss_tot)


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Plain R^2 of each target column (diagnostic only, not the competition score)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {}
    for j, name in enumerate(TARGETS):
        ss_res = np.sum((y_true[:, j] - y_pred[:, j]) ** 2)
        ss_tot = np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2)
        out[name] = float(1.0 - ss_res / ss_tot)
    return out
