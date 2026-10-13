import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retail_forecast.data import Dataset


def make_dataset(days=160, horizon=16, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=days + horizon)
    families = ["GROCERY I", "PRODUCE", "BEVERAGES"]
    rows = []
    for day, date in enumerate(dates):
        for store in [1, 2]:
            for family_idx, family in enumerate(families):
                promo = int((day + store + family_idx) % 9 == 0) * 8
                sales = max(
                    0,
                    20
                    + 12 * family_idx
                    + 8 * store
                    + 0.03 * day
                    + 7 * (date.dayofweek >= 5)
                    + 4 * promo
                    + rng.normal(0, 1),
                )
                rows.append((len(rows), date, store, family, sales, promo))
    panel = pd.DataFrame(
        rows, columns=["id", "date", "store_nbr", "family", "sales", "onpromotion"]
    )
    stores = pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["A", "D"],
            "cluster": [1, 2],
        }
    )
    holidays = pd.DataFrame(
        {
            "date": [dates[10], dates[11], dates[15], dates[18], dates[19]],
            "type": ["Holiday", "Transfer", "Holiday", "Holiday", "Event"],
            "locale": ["National", "National", "Local", "Regional", "National"],
            "locale_name": ["Ecuador", "Ecuador", "Quito", "Guayas", "Ecuador"],
            "description": [
                "Moved holiday",
                "Observed holiday",
                "City holiday",
                "State holiday",
                "Event",
            ],
            "transferred": [True, False, False, False, False],
        }
    )
    oil = pd.DataFrame({"date": dates[::2], "dcoilwtico": 40 + np.arange(len(dates[::2])) * 0.1})
    tx = pd.DataFrame(
        [(d, s, 100) for d in dates[:days] for s in [1, 2]],
        columns=["date", "store_nbr", "transactions"],
    )
    return Dataset.from_frames(
        train=panel[panel.date < dates[days]].copy(),
        test=panel[panel.date >= dates[days]].drop(columns="sales").copy(),
        stores=stores,
        oil=oil,
        holidays_events=holidays,
        transactions=tx,
    )


def write_dataset(data, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in vars(data).items():
        frame.to_csv(directory / f"{name}.csv", index=False)


@pytest.fixture
def dataset():
    return make_dataset()


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    from pyspark.sql import SparkSession

    path = tmp_path_factory.mktemp("spark")
    builder = (
        SparkSession.builder.master("local[2]")
        .appName("retail-forecast-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.warehouse.dir", str(path / "warehouse"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.sources.useV1SourceList", "avro,csv,json,kafka,orc,parquet,text,delta")
        .config("spark.databricks.delta.snapshotPartitions", "2")
    )
    jars = os.environ.get("RETAIL_DELTA_JARS")
    if not jars:
        raise RuntimeError("Use bash scripts/test.sh to prepare verified Delta jars before tests")
    session = (
        builder.config("spark.jars", jars)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
