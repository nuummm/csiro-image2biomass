"""Print the facts the CV design depends on: group counts and the additive identities.

uv run python scripts/inspect_data.py
"""

from __future__ import annotations

import argparse

import numpy as np

from biomass.config import load_config
from biomass.constraints import constraint_residuals
from biomass.data import TARGETS, load_test, load_train, targets_matrix
from biomass.folds import make_groups


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)

    train = load_train(cfg.get("data_dir"))
    test = load_test(cfg.get("data_dir"))
    print(f"train images: {len(train)}   test images (visible): {len(test)}")
    for scheme in ("group_date", "group_month_state"):
        groups = make_groups(train, scheme)
        _, counts = np.unique(groups, return_counts=True)
        print(
            f"{scheme:>18}: {len(counts)} groups, images per group "
            f"min {counts.min()} / median {int(np.median(counts))} / max {counts.max()}"
        )
    print("images per State:", train["State"].value_counts().to_dict())

    Y = targets_matrix(train)
    res = np.abs(constraint_residuals(Y))
    print(f"|GDM - (Green + Clover)|  max {res[:, 0].max():.4f}  mean {res[:, 0].mean():.4f}")
    print(f"|Total - (GDM + Dead)|    max {res[:, 1].max():.4f}  mean {res[:, 1].mean():.4f}")
    for j, t in enumerate(TARGETS):
        col = Y[:, j]
        print(
            f"{t:>13}: mean {col.mean():8.2f}  median {np.median(col):8.2f}  "
            f"max {col.max():8.2f}  zeros {np.mean(col == 0):.1%}"
        )


if __name__ == "__main__":
    main()
