"""Output formulations that relate the five targets.

The targets are not independent: GDM = Green + Clover and Total = GDM + Dead,
so all five are determined by the three non-negative components
``c = (Green, Dead, Clover)`` through ``y = A c``.

Four ways to produce the five outputs are compared:

* ``independent`` - fit all five, only clip at zero (constraints not enforced).
* ``sum``         - fit the three components, derive GDM and Total by adding.
* ``diff``        - fit Total, GDM and Green, derive Clover and Dead by subtracting.
* ``nnls``        - fit all five, then project onto ``{A c : c >= 0}`` with a
                    weighted non-negative least-squares problem per image.

Every formulation works on the raw (gram) scale, so it must be applied after
the inverse of any target transform (e.g. ``expm1`` after ``log1p``).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from biomass.data import CLOVER, DEAD, GDM, GREEN, TARGETS, TOTAL, weight_vector

COMPONENTS = (GREEN, DEAD, CLOVER)

# y = A @ c with rows in TARGETS order and columns in COMPONENTS order.
A = np.array(
    [
        [1.0, 0.0, 0.0],  # Green
        [0.0, 1.0, 0.0],  # Dead
        [0.0, 0.0, 1.0],  # Clover
        [1.0, 0.0, 1.0],  # GDM
        [1.0, 1.0, 1.0],  # Total
    ]
)

FITTED_TARGETS: dict[str, tuple[str, ...]] = {
    "independent": TARGETS,
    "sum": COMPONENTS,
    "diff": (TOTAL, GDM, GREEN),
    "nnls": TARGETS,
}
FORMULATIONS = tuple(FITTED_TARGETS)

_IDX = {t: i for i, t in enumerate(TARGETS)}


def fitted_columns(formulation: str) -> list[int]:
    """Column indices (in ``TARGETS`` order) that a formulation trains a model for."""
    return [_IDX[t] for t in FITTED_TARGETS[formulation]]


def compose(formulation: str, fitted: np.ndarray, nnls_weights: str = "metric") -> np.ndarray:
    """Map raw-scale predictions of the fitted targets to a full ``(n, 5)`` matrix."""
    fitted = np.asarray(fitted, dtype=float)
    n_fit = len(FITTED_TARGETS[formulation])
    if fitted.ndim != 2 or fitted.shape[1] != n_fit:
        raise ValueError(f"{formulation!r} expects {n_fit} columns, got {fitted.shape}")

    if formulation == "independent":
        return np.clip(fitted, 0.0, None)
    if formulation == "sum":
        return np.clip(fitted, 0.0, None) @ A.T
    if formulation == "diff":
        return _compose_diff(fitted)
    if formulation == "nnls":
        return project_nnls(fitted, weights=nnls_weights)
    raise ValueError(f"unknown formulation: {formulation!r}")


def _compose_diff(fitted: np.ndarray) -> np.ndarray:
    total, gdm, green = (np.clip(fitted[:, k], 0.0, None) for k in range(3))
    # A negative difference is set to zero and the parent is rebuilt from its
    # parts, so that the additive identities still hold exactly.
    clover = np.clip(gdm - green, 0.0, None)
    dead = np.clip(total - (green + clover), 0.0, None)
    return np.column_stack([green, dead, clover]) @ A.T


def project_nnls(y_hat: np.ndarray, weights: str | np.ndarray = "metric") -> np.ndarray:
    """Solve ``min_c ||W^(1/2) (A c - y_hat)||^2  s.t. c >= 0`` for each row.

    With ``weights="metric"`` the per-target weights equal the competition
    weights, so the projection minimises the same weighted squared error that
    enters the metric's residual sum of squares.
    """
    y_hat = np.asarray(y_hat, dtype=float)
    if isinstance(weights, str):
        if weights == "metric":
            w = weight_vector()
        elif weights == "identity":
            w = np.ones(len(TARGETS))
        else:
            raise ValueError(f"unknown nnls weights: {weights!r}")
    else:
        w = np.asarray(weights, dtype=float)
    sqrt_w = np.sqrt(w)
    a_w = A * sqrt_w[:, None]

    comps = np.empty((y_hat.shape[0], A.shape[1]))
    for i, row in enumerate(y_hat):
        comps[i], _ = nnls(a_w, row * sqrt_w)
    return comps @ A.T


def constraint_residuals(y: np.ndarray) -> np.ndarray:
    """``(n, 2)`` residuals of GDM - (Green + Clover) and Total - (GDM + Dead)."""
    y = np.asarray(y, dtype=float)
    gdm_res = y[:, _IDX[GDM]] - (y[:, _IDX[GREEN]] + y[:, _IDX[CLOVER]])
    total_res = y[:, _IDX[TOTAL]] - (y[:, _IDX[GDM]] + y[:, _IDX[DEAD]])
    return np.column_stack([gdm_res, total_res])
