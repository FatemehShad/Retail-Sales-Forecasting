import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import write_dataset

from retail_forecast.features import FeatureBuilder
from retail_forecast.model import ForecastModel, ModelConfig, fit, rmsle, train_and_evaluate


def test_rmsle_known_values_and_invalid_inputs():
    assert rmsle([0, 3], [0, 1]) == pytest.approx(np.log(2) / np.sqrt(2))
    for actual, pred in [([], []), ([1], [1, 2]), ([-1], [0]), ([1], [np.inf])]:
        with pytest.raises(ValueError):
            rmsle(actual, pred)


def test_weekly_baseline_is_recursive(dataset):
    model = ForecastModel(FeatureBuilder.from_dataset(dataset), None, "weekly_naive")
    predictions = model.predict(dataset.train, dataset.test)
    matrix = predictions.pivot(index="date", columns=["store_nbr", "family"], values="sales")
    np.testing.assert_allclose(matrix.iloc[0], matrix.iloc[7])
    np.testing.assert_allclose(matrix.iloc[0], matrix.iloc[14])
    assert len(predictions) == len(dataset.test)


def test_model_ignores_future_target_and_oil_values(dataset):
    config = ModelConfig(n_estimators=25, num_leaves=15, min_child_samples=5, n_jobs=1)
    model = fit(dataset, dataset.train, config)
    before = model.predict(dataset.train, dataset.test)
    future = dataset.test.copy()
    future["sales"] = 999999
    model.builder.oil = model.builder.oil.copy()
    model.builder.oil.loc[model.builder.oil.date > dataset.train.date.max(), "dcoilwtico"] = 999999
    after = model.predict(dataset.train, future)
    pd.testing.assert_frame_equal(before, after)


def test_training_backtests_selects_winner_and_serializes(dataset, tmp_path):
    config = ModelConfig(n_estimators=60, num_leaves=15, min_child_samples=5, n_jobs=1)
    model, metadata, backtests = train_and_evaluate(dataset, config)
    assert len(metadata["folds"]) == 6
    assert all(f["rows"] == 16 * 6 for f in metadata["folds"])
    assert all(f["cutoff"] < f["start"] <= f["end"] for f in metadata["folds"])
    assert len(backtests) == 3 * 2 * 16 * 6
    scores = metadata["mean_rmsle"]
    expected = "lightgbm" if scores["lightgbm"] < scores["weekly_naive"] else "weekly_naive"
    assert model.model_name == expected
    assert all(np.isfinite(list(scores.values())))
    model.save(tmp_path, metadata)
    loaded = ForecastModel.load(tmp_path)
    pd.testing.assert_frame_equal(
        model.predict(dataset.train, dataset.test), loaded.predict(dataset.train, dataset.test)
    )
    assert json.loads((tmp_path / "metadata.json").read_text())["selected_model"] == expected


def test_short_history_fails_instead_of_skipping_validation(dataset):
    dataset.train = dataset.train[
        dataset.train.date > dataset.train.date.max() - pd.Timedelta(days=60)
    ]
    with pytest.raises(ValueError, match="not enough history"):
        train_and_evaluate(dataset, ModelConfig())


def test_prediction_rejects_missing_and_unseen_series(dataset):
    model = ForecastModel(FeatureBuilder.from_dataset(dataset), None, "weekly_naive")
    with pytest.raises(ValueError, match="incomplete or unseen"):
        model.predict(dataset.train, dataset.test.iloc[1:])


def test_spark_date_objects_supported(dataset):
    model = ForecastModel(FeatureBuilder.from_dataset(dataset), None, "weekly_naive")
    history, future = dataset.train.copy(), dataset.test.copy()
    history["date"], future["date"] = history.date.dt.date, future.date.dt.date
    pd.testing.assert_frame_equal(
        model.predict(history, future), model.predict(dataset.train, dataset.test)
    )


def test_cli_end_to_end_train_and_predict(dataset, tmp_path):
    data, output, scored = tmp_path / "data", tmp_path / "output", tmp_path / "scored"
    write_dataset(dataset, data)
    command = [sys.executable, "-m", "retail_forecast.cli"]
    subprocess.run(
        [
            *command,
            "train",
            "--data-dir",
            str(data),
            "--output-dir",
            str(output),
            "--trees",
            "30",
            "--threads",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            *command,
            "predict",
            "--data-dir",
            str(data),
            "--output-dir",
            str(scored),
            "--model-dir",
            str(output / "model"),
        ],
        check=True,
        capture_output=True,
    )
    submission = pd.read_csv(output / "submission.csv")
    assert list(submission.columns) == ["id", "sales"]
    assert set(submission.id) == set(dataset.test.id)
    assert (submission.sales >= 0).all() and np.isfinite(submission.sales).all()
    pd.testing.assert_frame_equal(submission, pd.read_csv(scored / "submission.csv"))
