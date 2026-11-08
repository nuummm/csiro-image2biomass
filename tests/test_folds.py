import numpy as np
import pandas as pd
import pytest

from biomass.folds import make_folds, make_groups, make_repeated_folds


@pytest.fixture
def meta():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2016-01-01", periods=17, freq="23D").strftime("%Y/%m/%d")
    n = 240
    return pd.DataFrame(
        {
            "Sampling_Date": rng.choice(dates, size=n),
            "State": rng.choice(["NSW", "Vic", "Tas", "WA"], size=n),
        }
    )


@pytest.mark.parametrize("scheme", ["group_date", "group_month_state"])
def test_groups_do_not_cross_folds(meta, scheme):
    folds = make_folds(meta, scheme, n_splits=5, seed=0)
    groups = make_groups(meta, scheme)
    per_group = pd.Series(folds).groupby(groups).nunique()
    assert (per_group == 1).all()


@pytest.mark.parametrize("scheme", ["kfold", "group_date", "group_month_state"])
def test_same_seed_same_split(meta, scheme):
    a = make_folds(meta, scheme, 5, seed=7)
    b = make_folds(meta, scheme, 5, seed=7)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("scheme", ["kfold", "group_date"])
def test_every_row_validated_exactly_once(meta, scheme):
    folds = make_folds(meta, scheme, 5, seed=0)
    assert folds.shape == (len(meta),)
    assert set(np.unique(folds)) == set(range(5))
    # One fold index per row means each row is in exactly one validation set.
    assert sum((folds == k).sum() for k in range(5)) == len(meta)


def test_seeds_change_the_grouped_split(meta):
    table = make_repeated_folds(meta, "group_date", 5, seeds=[0, 1, 2])
    assert list(table.columns) == ["fold_s0", "fold_s1", "fold_s2"]
    assert not (table["fold_s0"] == table["fold_s1"]).all()


def test_too_few_groups_raises():
    meta = pd.DataFrame({"Sampling_Date": ["2016/01/01"] * 5 + ["2016/02/01"] * 5})
    with pytest.raises(ValueError):
        make_folds(meta, "group_date", n_splits=5, seed=0)
