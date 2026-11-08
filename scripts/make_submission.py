"""Step 3: write submission.csv from one run, or from an OOF-weighted ensemble of runs.

uv run python scripts/make_submission.py --runs runs/<name>
uv run python scripts/make_submission.py --runs runs/a runs/b --method crossfit_nnls
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from biomass.config import load_config
from biomass.data import load_train, resolve_data_dir, targets_matrix
from biomass.ensemble import METHODS, combine, ensemble_oof
from biomass.pipeline import load_run
from biomass.submit import make_submission, validate_submission


def ensemble_runs(runs: list[dict], y: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    """Weights are learned per seed column and averaged; the OOF score is reported per seed."""
    cols = list(runs[0]["folds"].columns)
    for r in runs[1:]:
        if not runs[0]["folds"].equals(r["folds"]) or r["image_ids"] != runs[0]["image_ids"]:
            raise ValueError(f"{r['dir']} uses different folds; ensemble needs identical CV")
    results = [
        ensemble_oof([r["oof"][c] for r in runs], y, runs[0]["folds"][c].to_numpy(), method)
        for c in cols
    ]
    weights = np.mean([res["weights"] for res in results], axis=0)
    test = combine([r["test"] for r in runs], weights)
    scores = [res["score"] for res in results]
    return test, {"weights": weights, "score_mean": np.mean(scores), "score_std": np.std(scores)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--method", choices=METHODS, default="mean")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    root = resolve_data_dir(cfg.get("data_dir"))
    test_long = pd.read_csv(root / "test.csv")
    runs = [load_run(p) for p in args.runs]
    for r in runs:
        if r["test"] is None or r["test_ids"] != runs[0]["test_ids"]:
            raise ValueError(f"{r['dir']} has no (or mismatched) test predictions")

    if len(runs) == 1:
        test = runs[0]["test"]
    else:
        train = load_train(cfg.get("data_dir"))
        if train["image_id"].tolist() != runs[0]["image_ids"]:
            raise ValueError("runs were trained on a different train.csv")
        test, info = ensemble_runs(runs, targets_matrix(train), args.method)
        print(
            f"ensemble[{args.method}] OOF {info['score_mean']:.4f} +/- {info['score_std']:.4f}"
            f" weights {np.round(info['weights'], 3).tolist()}"
        )

    sub = make_submission(test_long, runs[0]["test_ids"], test)
    stats = validate_submission(sub, test_long)
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {stats}")


if __name__ == "__main__":
    main()
