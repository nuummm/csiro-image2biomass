"""Glue used by the scripts: embedding cache, one CV experiment, run directories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from biomass.cv import CVConfig, run_repeated
from biomass.data import load_test, load_train, targets_matrix
from biomass.folds import make_groups, make_repeated_folds

# --------------------------------------------------------------------------- embeddings


def emb_file(emb_dir: str | Path, backbone: str, split: str) -> Path:
    return Path(emb_dir) / backbone / f"{split}.npy"


def save_embeddings(emb_dir, backbone: str, split: str, ids, X: np.ndarray, meta: dict) -> Path:
    path = emb_file(emb_dir, backbone, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, X.astype(np.float32))
    path.with_name(f"{split}_ids.txt").write_text("\n".join(ids) + "\n")
    (path.parent / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def load_embeddings(emb_dir, backbone: str, split: str, ids) -> np.ndarray:
    """Load cached features and reorder them to match ``ids``."""
    path = emb_file(emb_dir, backbone, split)
    X = np.load(path)
    cached = path.with_name(f"{split}_ids.txt").read_text().split()
    pos = {img: i for i, img in enumerate(cached)}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise KeyError(f"{len(missing)} images missing from {path}, e.g. {missing[:3]}")
    return X[[pos[i] for i in ids]]


# --------------------------------------------------------------------------- experiments


def run_name(cfg: dict) -> str:
    m = cfg["model"]
    return "-".join(
        [cfg["backbone"], cfg["cv"]["scheme"], m["head"], m["target_space"], m["formulation"]]
    )


def cv_config(cfg: dict) -> CVConfig:
    return CVConfig(**cfg["model"])


def run_experiment(cfg: dict, name: str | None = None, with_test: bool = True) -> Path:
    """Train with the CV in ``cfg`` and write everything to ``runs_dir/<name>/``."""
    train = load_train(cfg.get("data_dir"))
    image_ids = train["image_id"].tolist()
    X = load_embeddings(cfg["emb_dir"], cfg["backbone"], "train", image_ids)
    Y = targets_matrix(train)

    test_ids, X_test = [], None
    if with_test and emb_file(cfg["emb_dir"], cfg["backbone"], "test").exists():
        test_ids = load_test(cfg.get("data_dir"))["image_id"].tolist()
        X_test = load_embeddings(cfg["emb_dir"], cfg["backbone"], "test", test_ids)

    cv = cfg["cv"]
    fold_table = make_repeated_folds(train, cv["scheme"], cv["n_splits"], cv["seeds"])
    groups = make_groups(train, cv["scheme"])
    result = run_repeated(X, Y, fold_table, cv_config(cfg), X_test, groups)

    run_dir = Path(cfg["runs_dir"]) / (name or run_name(cfg))
    save_run(run_dir, cfg, result, image_ids, test_ids, fold_table)
    return run_dir


def save_run(run_dir: Path, cfg: dict, result: dict, image_ids, test_ids, fold_table) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    fold_table.assign(image_id=image_ids).to_csv(run_dir / "folds.csv", index=False)
    for col, oof in result["oof"].items():
        np.save(run_dir / f"oof_{col}.npy", oof)
    if result["test"] is not None:
        np.save(run_dir / "test.npy", result["test"])
        (run_dir / "test_ids.txt").write_text("\n".join(test_ids) + "\n")


def is_current(run_dir: str | Path, cfg: dict, image_ids: list[str]) -> bool:
    """True if ``run_dir`` holds results for exactly this config and training set.

    The run name only encodes the main axes, so seeds, fold count, PCA size or a
    different dataset would otherwise silently reuse an old run.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "summary.json").exists():
        return False
    if json.loads((run_dir / "config.json").read_text()) != json.loads(json.dumps(cfg)):
        return False
    saved_ids = pd.read_csv(run_dir / "folds.csv", usecols=["image_id"])["image_id"].tolist()
    return saved_ids == list(image_ids)


def load_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    folds = pd.read_csv(run_dir / "folds.csv")
    fold_cols = [c for c in folds.columns if c.startswith("fold_")]
    out = {
        "dir": run_dir,
        "config": json.loads((run_dir / "config.json").read_text()),
        "summary": json.loads((run_dir / "summary.json").read_text()),
        "image_ids": folds["image_id"].tolist(),
        "folds": folds[fold_cols],
        "oof": {c: np.load(run_dir / f"oof_{c}.npy") for c in fold_cols},
        "test": None,
        "test_ids": [],
    }
    if (run_dir / "test.npy").exists():
        out["test"] = np.load(run_dir / "test.npy")
        out["test_ids"] = (run_dir / "test_ids.txt").read_text().split()
    return out
