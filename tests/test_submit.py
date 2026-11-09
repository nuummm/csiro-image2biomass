import numpy as np
import pandas as pd
import pytest

from biomass.constraints import A
from biomass.data import TARGETS, make_sample_id, split_sample_id
from biomass.submit import make_submission, validate_submission


@pytest.fixture
def test_long():
    ids = [f"ID{i}" for i in (3, 1, 2)]
    rows = [
        {"sample_id": make_sample_id(i, t), "image_path": f"test/{i}.jpg", "target_name": t}
        for i in ids
        for t in TARGETS
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def preds():
    rng = np.random.default_rng(0)
    return rng.gamma(2.0, 10.0, size=(3, 3)) @ A.T


def test_five_rows_per_image_in_test_order(test_long, preds):
    sub = make_submission(test_long, ["ID1", "ID2", "ID3"], preds)
    assert list(sub.columns) == ["sample_id", "target"]
    assert len(sub) == 15
    assert sub["sample_id"].tolist() == test_long["sample_id"].tolist()
    assert sub.groupby(sub["sample_id"].str.split("__").str[0]).size().eq(5).all()


def test_values_land_on_the_right_ids(test_long, preds):
    sub = make_submission(test_long, ["ID1", "ID2", "ID3"], preds).set_index("sample_id")
    for i, img in enumerate(["ID1", "ID2", "ID3"]):
        for j, t in enumerate(TARGETS):
            assert sub.loc[make_sample_id(img, t), "target"] == pytest.approx(preds[i, j])


def test_stats_report_constraint_residual(test_long, preds):
    sub = make_submission(test_long, ["ID1", "ID2", "ID3"], preds)
    stats = validate_submission(sub, test_long)
    assert stats["n_images"] == 3 and stats["n_rows"] == 15
    assert stats["max_constraint_residual"] < 1e-9


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda s: s.assign(target=s["target"].where(s.index != 0, np.nan)),
        lambda s: s.assign(target=s["target"].where(s.index != 0, -1.0)),
        lambda s: pd.concat([s, s.iloc[[0]]], ignore_index=True),
        lambda s: s.iloc[1:].reset_index(drop=True),
    ],
    ids=["nan", "negative", "duplicate", "missing"],
)
def test_invalid_submissions_are_rejected(test_long, preds, corrupt):
    sub = make_submission(test_long, ["ID1", "ID2", "ID3"], preds)
    with pytest.raises(ValueError):
        validate_submission(corrupt(sub), test_long)


def test_identity_check_is_opt_in(test_long, preds):
    broken = preds.copy()
    broken[0, 3] += 1.0  # GDM no longer equals Green + Clover
    sub = make_submission(test_long, ["ID1", "ID2", "ID3"], broken)
    assert validate_submission(sub, test_long)["max_constraint_residual"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="identities"):
        validate_submission(sub, test_long, identity_tol=1e-6)


def test_missing_image_prediction_is_rejected(test_long, preds):
    with pytest.raises(ValueError):
        make_submission(test_long, ["ID1", "ID2", "ID9"], preds)


def test_sample_id_round_trip():
    assert split_sample_id(make_sample_id("ID1001187975", "GDM_g")) == ("ID1001187975", "GDM_g")
    with pytest.raises(ValueError):
        split_sample_id("ID1__Height")
