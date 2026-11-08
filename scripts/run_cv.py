"""Step 2: cross-validate one configuration and save OOF / test predictions.

uv run python scripts/run_cv.py
uv run python scripts/run_cv.py --set cv.scheme=kfold --set model.head=lgbm
"""

from __future__ import annotations

import argparse
import json

from biomass.config import load_config
from biomass.pipeline import run_experiment


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. model.head=lgbm")
    ap.add_argument("--name", default=None, help="run directory name (default: from config)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    run_dir = run_experiment(cfg, name=args.name)
    summary = json.loads((run_dir / "summary.json").read_text())
    print(f"{run_dir}")
    print(
        f"  OOF weighted R2 {summary['oof_score_mean']:.4f} +/- {summary['oof_score_std']:.4f}"
        f" (over seeds); per fold {summary['fold_score_mean']:.4f}"
        f" +/- {summary['fold_score_std']:.4f}"
    )


if __name__ == "__main__":
    main()
