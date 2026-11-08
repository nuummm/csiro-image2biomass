"""Loading the competition CSVs and converting between long and wide layouts.

The files are in *long* format: one row per (image, target) pair, with
``sample_id = "<image_id>__<target_name>"``. Modelling is easier in *wide*
format: one row per image and one column per target.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

GREEN, DEAD, CLOVER, GDM, TOTAL = (
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
)
TARGETS: tuple[str, ...] = (GREEN, DEAD, CLOVER, GDM, TOTAL)

# Per-row weights of the official metric (Evaluation page).
TARGET_WEIGHTS: dict[str, float] = {GREEN: 0.1, DEAD: 0.1, CLOVER: 0.1, GDM: 0.2, TOTAL: 0.5}

# Columns that exist only in train.csv. They are kept for EDA / grouping but
# never used as model inputs because test.csv does not have them.
TRAIN_ONLY_COLUMNS = ("Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm")

SEP = "__"
DATA_DIR_ENV = "BIOMASS_DATA_DIR"


def weight_vector() -> np.ndarray:
    return np.array([TARGET_WEIGHTS[t] for t in TARGETS])


def make_sample_id(image_id: str, target: str) -> str:
    return f"{image_id}{SEP}{target}"


def split_sample_id(sample_id: str) -> tuple[str, str]:
    image_id, target = sample_id.rsplit(SEP, 1)
    if target not in TARGETS:
        raise ValueError(f"unknown target in sample_id: {sample_id!r}")
    return image_id, target


def resolve_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    value = data_dir if data_dir is not None else os.environ.get(DATA_DIR_ENV)
    if not value:
        raise ValueError(f"pass --data-dir or set ${DATA_DIR_ENV}")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"data directory not found: {path}")
    return path


def _image_id_from_path(image_path: str) -> str:
    return Path(image_path).stem


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long frame (with ``target_name`` / ``target``) to one row per image."""
    df = long_df.copy()
    df["image_id"] = df["image_path"].map(_image_id_from_path)
    dup = df.duplicated(["image_id", "target_name"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated (image, target) rows")

    wide = df.pivot(index="image_id", columns="target_name", values="target")
    missing = [t for t in TARGETS if t not in wide.columns]
    if missing:
        raise ValueError(f"targets missing from file: {missing}")
    if wide[list(TARGETS)].isna().any().any():
        raise ValueError("some images do not have all five targets")
    wide = wide[list(TARGETS)]

    meta_cols = ["image_path", *[c for c in TRAIN_ONLY_COLUMNS if c in df.columns]]
    meta = df.groupby("image_id", sort=True)[meta_cols].first()
    return meta.join(wide).reset_index()


def load_train(data_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    """Return train.csv in wide format (one row per image, sorted by image_id)."""
    root = resolve_data_dir(data_dir)
    wide = long_to_wide(pd.read_csv(root / "train.csv"))
    wide["image_path"] = [str(resolve_image(root, p, "train")) for p in wide["image_path"]]
    return wide


def load_test(data_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    """Return the unique test images (``image_id``, ``image_path``) and the raw long frame."""
    root = resolve_data_dir(data_dir)
    long_df = pd.read_csv(root / "test.csv")
    images = (
        long_df.assign(image_id=long_df["image_path"].map(_image_id_from_path))
        .drop_duplicates("image_id")[["image_id", "image_path"]]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    images["image_path"] = [str(resolve_image(root, p, "test")) for p in images["image_path"]]
    return images


def resolve_image(root: Path, image_path: str, split: str) -> Path:
    """Resolve a CSV image path relative to the data dir, falling back to ``<split>/<name>``."""
    candidate = root / image_path
    if candidate.exists():
        return candidate
    return root / split / Path(image_path).name


def targets_matrix(wide: pd.DataFrame) -> np.ndarray:
    return wide[list(TARGETS)].to_numpy(dtype=float)


def wide_to_long(image_ids: list[str] | np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    """Turn an ``(n_images, 5)`` prediction matrix into the submission layout."""
    preds = np.asarray(preds, dtype=float)
    if preds.shape != (len(image_ids), len(TARGETS)):
        raise ValueError(f"expected shape ({len(image_ids)}, {len(TARGETS)}), got {preds.shape}")
    rows = [
        (make_sample_id(img, t), preds[i, j])
        for i, img in enumerate(image_ids)
        for j, t in enumerate(TARGETS)
    ]
    return pd.DataFrame(rows, columns=["sample_id", "target"])
