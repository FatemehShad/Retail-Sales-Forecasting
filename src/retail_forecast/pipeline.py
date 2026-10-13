"""Spark CSV contracts and portable SQL transformations used by the Databricks jobs."""

import re

from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

CSV_SCHEMAS = {
    "train": [
        ("id", LongType()),
        ("date", DateType()),
        ("store_nbr", IntegerType()),
        ("family", StringType()),
        ("sales", DoubleType()),
        ("onpromotion", IntegerType()),
    ],
    "test": [
        ("id", LongType()),
        ("date", DateType()),
        ("store_nbr", IntegerType()),
        ("family", StringType()),
        ("onpromotion", IntegerType()),
    ],
    "stores": [
        ("store_nbr", IntegerType()),
        ("city", StringType()),
        ("state", StringType()),
        ("type", StringType()),
        ("cluster", IntegerType()),
    ],
    "oil": [("date", DateType()), ("dcoilwtico", DoubleType())],
    "holidays_events": [
        ("date", DateType()),
        ("type", StringType()),
        ("locale", StringType()),
        ("locale_name", StringType()),
        ("description", StringType()),
        ("transferred", BooleanType()),
    ],
    "transactions": [
        ("date", DateType()),
        ("store_nbr", IntegerType()),
        ("transactions", LongType()),
    ],
}
UNIQUE_KEYS = {
    "train": ["date", "store_nbr", "family"],
    "test": ["date", "store_nbr", "family"],
    "stores": ["store_nbr"],
    "oil": ["date"],
    "transactions": ["date", "store_nbr"],
}


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return f"`{value}`"


def namespace(catalog: str, schema: str) -> str:
    return f"{identifier(catalog)}.{identifier(schema)}"


def read_validated_csv(spark, path: str, name: str):
    fields = [StructField(column, dtype, True) for column, dtype in CSV_SCHEMAS[name]]
    frame = (
        spark.read.schema(StructType(fields))
        .option("header", True)
        .option("enforceSchema", False)
        .option("mode", "FAILFAST")
        .option("dateFormat", "yyyy-MM-dd")
        .csv(path)
    )
    # Materialize while FAILFAST/header checks still apply; never silently drop bad rows.
    if frame.count() == 0:
        raise ValueError(f"{name}: empty source")
    required = [column for column, _ in CSV_SCHEMAS[name] if column != "dcoilwtico"]
    bad = F.lit(False)
    for column in required:
        bad = bad | F.col(column).isNull()
    for column in ["sales", "onpromotion", "transactions"]:
        if column in frame.columns:
            bad = bad | (F.col(column) < 0) | F.isnan(column) | (F.abs(F.col(column)) > 1e308)
    if "dcoilwtico" in frame.columns:
        bad = bad | F.isnan("dcoilwtico") | (F.abs(F.col("dcoilwtico")) > 1e308)
    if name == "holidays_events":
        bad = bad | ~F.col("locale").isin("National", "Regional", "Local")
    if frame.filter(bad).limit(1).count():
        raise ValueError(f"{name}: null keys/metadata, invalid locale, or invalid numeric values")
    key = UNIQUE_KEYS.get(name)
    if key and frame.groupBy(*key).count().filter("count > 1").limit(1).count():
        raise ValueError(f"{name}: duplicate keys")
    if "id" in frame.columns and frame.groupBy("id").count().filter("count > 1").limit(1).count():
        raise ValueError(f"{name}: duplicate ids")
    return frame


def validate_panel(spark, tables: dict) -> None:
    train, test, stores = tables["train"], tables["test"], tables["stores"]
    for name, frame in [("train", train), ("test", test)]:
        if (
            frame.select("store_nbr")
            .distinct()
            .join(stores, "store_nbr", "left_anti")
            .limit(1)
            .count()
        ):
            raise ValueError(f"{name}: store metadata missing")
    if train.select("id").join(test.select("id"), "id").limit(1).count():
        raise ValueError("train/test ids overlap")
    bounds = train.agg(F.max("date").alias("last")).first()
    dates = test.select("date").distinct().orderBy("date").collect()
    from datetime import timedelta

    if dates[0].date != bounds.last + timedelta(days=1):
        raise ValueError("test must begin the day after training ends")
    if dates[-1].date - dates[0].date != timedelta(days=len(dates) - 1):
        raise ValueError("test horizon must contain consecutive calendar days")
    pairs = train.select("store_nbr", "family").distinct()
    expected = pairs.crossJoin(test.select("date").distinct())
    keys = ["date", "store_nbr", "family"]
    if (
        expected.join(test, keys, "left_anti").limit(1).count()
        or test.join(expected, keys, "left_anti").limit(1).count()
    ):
        raise ValueError("test contains incomplete or unseen series")


def silver_queries(prefix: str) -> dict[str, str]:
    """All target lags live in Python; SQL analytics never feed future actuals to the model."""
    return {
        "silver_train": (
            f"SELECT id, date, store_nbr, family, sales, onpromotion FROM {prefix}.bronze_train"
        ),
        "silver_test": (
            f"SELECT id, date, store_nbr, family, onpromotion FROM {prefix}.bronze_test"
        ),
        "silver_stores": (
            f"SELECT store_nbr, city, state, type, cluster FROM {prefix}.bronze_stores"
        ),
        "silver_oil": f"SELECT date, dcoilwtico FROM {prefix}.bronze_oil",
        "silver_holidays": f"SELECT * FROM {prefix}.bronze_holidays_events",
        "silver_transactions": f"SELECT * FROM {prefix}.bronze_transactions",
    }


def analytics_queries(prefix: str) -> dict[str, str]:
    return {
        "gold_store_daily_sales": f"""
            SELECT t.date, t.store_nbr, s.city, s.state,
                   SUM(t.sales) AS sales, SUM(t.onpromotion) AS promoted_items,
                   COUNT(DISTINCT t.family) AS families
            FROM {prefix}.silver_train t
            JOIN {prefix}.silver_stores s ON t.store_nbr = s.store_nbr
            GROUP BY t.date, t.store_nbr, s.city, s.state
        """,
        "gold_family_daily_sales": f"""
            SELECT date, family, SUM(sales) AS sales, SUM(onpromotion) AS promoted_items
            FROM {prefix}.silver_train GROUP BY date, family
        """,
        "gold_store_transactions": f"""
            SELECT d.*, x.transactions,
                   CASE WHEN x.transactions > 0 THEN d.sales / x.transactions
                        END AS units_per_transaction
            FROM {prefix}.gold_store_daily_sales d
            LEFT JOIN {prefix}.silver_transactions x
              ON d.date = x.date AND d.store_nbr = x.store_nbr
        """,
    }
