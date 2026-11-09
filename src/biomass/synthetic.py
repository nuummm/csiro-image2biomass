"""A tiny fake dataset in the competition's file layout, for tests and dry runs.

Images are random noise whose share of green / brown pixels follows the
synthetic biomass values, so that even simple colour features carry signal.
Each sampling date also gets a colour cast and a hidden scale factor on the
recorded targets: the cast identifies the date, the factor is not visible in
the pixel shares, so a random KFold can learn it from same-date neighbours
while a date-grouped CV cannot. The numbers have nothing to do with the real
data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from biomass.constraints import A
from biomass.data import TARGETS, make_sample_id


def _image(rng: np.random.Generator, green: float, dead: float, clover: float, size, cast):
    w, h = size
    total = green + dead + clover
    p_green = (green + clover) / (total + 20.0)
    p_dead = dead / (total + 20.0)
    u = rng.random((h, w))
    img = np.empty((h, w, 3))
    img[:] = (0.35, 0.28, 0.20)  # soil
    img[u < p_green + p_dead] = (0.62, 0.55, 0.35)  # dead material
    img[u < p_green] = (0.20, 0.55, 0.18)  # green material
    img += np.asarray(cast) + rng.normal(0, 0.04, img.shape)
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def make_synthetic_dataset(
    root: str | Path,
    n_train: int = 60,
    n_test: int = 8,
    n_dates: int = 10,
    image_size: tuple[int, int] = (128, 64),
    seed: int = 0,
) -> Path:
    """Write ``train.csv``, ``test.csv``, ``sample_submission.csv`` and JPEGs under ``root``."""
    root = Path(root)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Ten days apart, two consecutive visits per site: months hold several dates
    # and month x State groups are coarser than dates.
    dates = pd.date_range("2015-01-15", periods=n_dates, freq="10D").strftime("%Y/%m/%d")
    states = ["StateA", "StateB", "StateC"]
    date_cast = rng.normal(0, 0.05, size=(n_dates, 3))
    date_factor = rng.uniform(0.6, 1.5, size=n_dates)

    train_rows = []
    for i in range(n_train):
        image_id = f"ID{1000000 + i}"
        date_idx = rng.integers(n_dates)
        scale = 0.5 + date_idx / n_dates  # date-level effect, so grouping matters
        comps = np.array(
            [
                rng.gamma(2.0, 15.0 * scale),
                rng.gamma(1.5, 10.0),
                rng.gamma(1.0, 5.0) * (rng.random() < 0.6),
            ]
        )
        y = A @ (comps * date_factor[date_idx])
        rel = f"train/{image_id}.jpg"
        _image(rng, *comps, image_size, date_cast[date_idx]).save(root / rel, quality=90)
        for t, v in zip(TARGETS, y, strict=True):
            train_rows.append(
                {
                    "sample_id": make_sample_id(image_id, t),
                    "image_path": rel,
                    "Sampling_Date": dates[date_idx],
                    "State": states[(date_idx // 2) % len(states)],
                    "Species": "Ryegrass_Clover",
                    "Pre_GSHH_NDVI": round(float(rng.uniform(0.3, 0.9)), 2),
                    "Height_Ave_cm": round(float(rng.uniform(1, 20)), 1),
                    "target_name": t,
                    "target": round(float(v), 4),
                }
            )
    pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)

    test_rows = []
    for i in range(n_test):
        image_id = f"ID{2000000 + i}"
        comps = np.array([rng.gamma(2.0, 15.0), rng.gamma(1.5, 10.0), rng.gamma(1.0, 5.0)])
        rel = f"test/{image_id}.jpg"
        cast = rng.normal(0, 0.05, size=3)
        _image(rng, *comps, image_size, cast).save(root / rel, quality=90)
        for t in TARGETS:
            test_rows.append(
                {"sample_id": make_sample_id(image_id, t), "image_path": rel, "target_name": t}
            )
    test = pd.DataFrame(test_rows)
    test.to_csv(root / "test.csv", index=False)
    test[["sample_id"]].assign(target=0.0).to_csv(root / "sample_submission.csv", index=False)
    return root
