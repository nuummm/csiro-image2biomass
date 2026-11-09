"""Writing and validating the long-format submission file."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biomass.constraints import constraint_residuals
from biomass.data import TARGETS, split_sample_id, wide_to_long


def make_submission(
    test_long: pd.DataFrame, image_ids, preds: np.ndarray, identity_tol: float | None = None
) -> pd.DataFrame:
    """Return ``sample_id, target`` rows in the order of ``test.csv``."""
    sub = wide_to_long(list(image_ids), preds)
    merged = test_long[["sample_id"]].merge(sub, on="sample_id", how="left", validate="1:1")
    validate_submission(merged, test_long, identity_tol)
    return merged


def validate_submission(
    sub: pd.DataFrame, test_long: pd.DataFrame, identity_tol: float | None = None
) -> dict:
    """Raise on any structural problem; return a few statistics otherwise.

    With ``identity_tol`` set, also raise when GDM or Total differ from the sum
    of their parts by more than that many grams.
    """
    if list(sub.columns) != ["sample_id", "target"]:
        raise ValueError(f"columns must be ['sample_id', 'target'], got {list(sub.columns)}")
    if sub["sample_id"].duplicated().any():
        raise ValueError("duplicated sample_id")
    if set(sub["sample_id"]) != set(test_long["sample_id"]):
        raise ValueError("sample_id set differs from test.csv")
    if sub["target"].isna().any() or not np.isfinite(sub["target"]).all():
        raise ValueError("NaN or infinite predictions")
    if (sub["target"] < 0).any():
        raise ValueError("negative predictions")

    parts = sub["sample_id"].map(split_sample_id)
    wide = sub.assign(image_id=[p[0] for p in parts], target_name=[p[1] for p in parts]).pivot(
        index="image_id", columns="target_name", values="target"
    )[list(TARGETS)]
    if wide.isna().any().any():
        raise ValueError("some images do not have exactly five rows")
    resid = np.abs(constraint_residuals(wide.to_numpy()))
    if identity_tol is not None and resid.max() > identity_tol:
        raise ValueError(f"additive identities violated by up to {resid.max():.3g} g")
    return {
        "n_images": len(wide),
        "n_rows": len(sub),
        "max_constraint_residual": float(resid.max()),
    }
