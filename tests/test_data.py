import pandas as pd
import pytest

from biomass.data import TARGETS, long_to_wide, make_sample_id


def _long(image_ids, targets=TARGETS):
    return pd.DataFrame(
        [
            {
                "sample_id": make_sample_id(i, t),
                "image_path": f"train/{i}.jpg",
                "Sampling_Date": "2015/01/15",
                "target_name": t,
                "target": float(k),
            }
            for i in image_ids
            for k, t in enumerate(targets)
        ]
    )


def test_one_row_per_image_with_targets_in_order():
    wide = long_to_wide(_long(["ID2", "ID1"]))
    assert wide["image_id"].tolist() == ["ID1", "ID2"]
    assert wide[list(TARGETS)].iloc[0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert "Sampling_Date" in wide.columns


@pytest.mark.parametrize(
    "frame",
    [
        lambda: pd.concat([_long(["ID1"]), _long(["ID1"]).iloc[[0]]]),
        lambda: _long(["ID1"], TARGETS[:4]),
        lambda: pd.concat([_long(["ID1"]), _long(["ID2"]).iloc[:4]]),
    ],
    ids=["duplicate", "missing_target_column", "incomplete_image"],
)
def test_malformed_files_are_rejected(frame):
    with pytest.raises(ValueError):
        long_to_wide(frame())
