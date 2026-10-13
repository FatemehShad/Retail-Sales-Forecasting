import numpy as np
import pandas as pd

from retail_forecast.features import FeatureBuilder


def test_target_features_do_not_see_current_or_future_sales(dataset):
    builder = FeatureBuilder.from_dataset(dataset)
    day = dataset.train.date.min() + pd.Timedelta(days=60)
    rows = dataset.train[dataset.train.date.eq(day)].copy()
    before = builder.transform(rows, dataset.train)
    changed = dataset.train.copy()
    changed.loc[changed.date >= day, "sales"] = 999999
    after = builder.transform(rows, changed)
    pd.testing.assert_frame_equal(before, after)
    previous = dataset.train[dataset.train.date.eq(day - pd.Timedelta(days=1))].sales.to_numpy()
    np.testing.assert_allclose(before.lag_1, np.log1p(previous), rtol=1e-6)


def test_missing_calendar_date_is_not_previous_observation(dataset):
    builder = FeatureBuilder.from_dataset(dataset)
    day = dataset.train.date.min() + pd.Timedelta(days=60)
    rows = dataset.train[dataset.train.date.eq(day)]
    history = dataset.train[~dataset.train.date.eq(day - pd.Timedelta(days=1))]
    features = builder.transform(rows, history)
    assert features.lag_1.isna().all()
    assert features.lag_7.notna().all()


def test_oil_forecast_freezes_at_origin(dataset):
    builder = FeatureBuilder.from_dataset(dataset)
    cutoff = dataset.train.date.max()
    rows = dataset.test.copy()
    before = builder.transform(rows, dataset.train, oil_cutoff=cutoff)
    builder.oil = builder.oil.copy()
    builder.oil.loc[builder.oil.date > cutoff, "dcoilwtico"] = 999999
    after = builder.transform(rows, dataset.train, oil_cutoff=cutoff)
    pd.testing.assert_series_equal(before.oil, after.oil)
    assert before.oil.nunique() == 1


def test_oil_is_never_backward_filled(dataset):
    builder = FeatureBuilder.from_dataset(dataset)
    day = dataset.train.date.min()
    result = builder.transform(dataset.train[dataset.train.date.eq(day)], dataset.train)
    assert result.oil.isna().all()


def test_holidays_respect_transfer_and_location(dataset):
    builder = FeatureBuilder.from_dataset(dataset)
    result = builder.transform(dataset.train, dataset.train)
    start = dataset.train.date.min()
    assert result.loc[dataset.train.date.eq(start + pd.Timedelta(days=10)), "holiday"].sum() == 0
    assert result.loc[dataset.train.date.eq(start + pd.Timedelta(days=11)), "holiday"].sum() == 6
    city = dataset.train.date.eq(start + pd.Timedelta(days=15))
    assert result.loc[city & dataset.train.store_nbr.eq(1), "holiday"].sum() == 3
    assert result.loc[city & dataset.train.store_nbr.eq(2), "holiday"].sum() == 0
    state = dataset.train.date.eq(start + pd.Timedelta(days=18))
    assert result.loc[state & dataset.train.store_nbr.eq(2), "holiday"].sum() == 3
