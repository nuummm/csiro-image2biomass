"""Cross-validation splits.

With a few hundred images taken on a limited number of dates and sites, images
from the same sampling event look alike. A random KFold puts near-duplicates on
both sides of the split and is expected to be optimistic; grouping by
``Sampling_Date`` keeps each event inside a single fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SCHEMES = ("kfold", "group_date", "group_month_state")


def make_groups(meta: pd.DataFrame, scheme: str) -> np.ndarray | None:
    if scheme == "kfold":
        return None
    if scheme == "group_date":
        return meta["Sampling_Date"].astype(str).to_numpy()
    if scheme == "group_month_state":
        month = pd.to_datetime(meta["Sampling_Date"]).dt.strftime("%Y-%m")
        return (month + "|" + meta["State"].astype(str)).to_numpy()
    raise ValueError(f"unknown CV scheme: {scheme!r}")


def assign_groups(groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Seeded group k-fold: visit groups in random order and put each into the smallest fold."""
    uniq, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(uniq) < n_splits:
        raise ValueError(f"only {len(uniq)} groups for {n_splits} folds")
    rng = np.random.default_rng(seed)
    fold_of_group = np.empty(len(uniq), dtype=int)
    fold_sizes = np.zeros(n_splits, dtype=int)
    for g in rng.permutation(len(uniq)):
        k = int(np.argmin(fold_sizes))
        fold_of_group[g] = k
        fold_sizes[k] += counts[g]
    return fold_of_group[inverse]


def make_folds(meta: pd.DataFrame, scheme: str, n_splits: int, seed: int) -> np.ndarray:
    """Return a fold index in ``[0, n_splits)`` for every row of ``meta``."""
    n = len(meta)
    groups = make_groups(meta, scheme)
    if groups is None:
        folds = np.empty(n, dtype=int)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (_, val_idx) in enumerate(splitter.split(np.arange(n))):
            folds[val_idx] = k
    else:
        folds = assign_groups(groups, n_splits, seed)
        check_no_group_overlap(folds, groups)
    if set(np.unique(folds)) != set(range(n_splits)):
        raise ValueError("some folds are empty")
    return folds


def check_no_group_overlap(folds: np.ndarray, groups: np.ndarray) -> None:
    per_group = pd.Series(folds).groupby(pd.Series(groups)).nunique()
    leaking = per_group[per_group > 1]
    if not leaking.empty:
        raise AssertionError(f"groups split across folds: {list(leaking.index)[:5]}")


def make_repeated_folds(
    meta: pd.DataFrame, scheme: str, n_splits: int, seeds: list[int]
) -> pd.DataFrame:
    """One column of fold indices per seed (``fold_s<seed>``), indexed like ``meta``."""
    return pd.DataFrame(
        {f"fold_s{s}": make_folds(meta, scheme, n_splits, s) for s in seeds}, index=meta.index
    )
