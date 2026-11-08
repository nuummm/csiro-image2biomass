"""Write a small fake dataset in the competition layout (no download needed).

uv run python scripts/make_synthetic_data.py --out data/synthetic
BIOMASS_DATA_DIR=data/synthetic uv run python scripts/run_all.py --set backbone=color_stats
"""

from __future__ import annotations

import argparse

from biomass.synthetic import make_synthetic_dataset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    root = make_synthetic_dataset(
        args.out, n_train=args.n_train, n_test=args.n_test, seed=args.seed
    )
    print(f"wrote synthetic dataset to {root}")


if __name__ == "__main__":
    main()
