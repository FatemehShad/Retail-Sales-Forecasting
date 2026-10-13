import pandas as pd
import pytest
from conftest import write_dataset
from py4j.protocol import Py4JJavaError

from retail_forecast.pipeline import (
    CSV_SCHEMAS,
    analytics_queries,
    namespace,
    read_validated_csv,
    silver_queries,
    validate_panel,
)


def test_ingestion_and_sql_analytics(spark, dataset, tmp_path):
    write_dataset(dataset, tmp_path)
    tables = {
        name: read_validated_csv(spark, str(tmp_path / f"{name}.csv"), name) for name in CSV_SCHEMAS
    }
    validate_panel(spark, tables)
    spark.sql("CREATE DATABASE IF NOT EXISTS retail_test")
    for name, frame in tables.items():
        frame.write.format("delta").mode("overwrite").saveAsTable(f"retail_test.bronze_{name}")
    for name, query in silver_queries("retail_test").items():
        spark.sql(query).write.format("delta").mode("overwrite").saveAsTable(f"retail_test.{name}")
    for name, query in analytics_queries("retail_test").items():
        spark.sql(query).write.format("delta").mode("overwrite").saveAsTable(f"retail_test.{name}")
    result = (
        spark.table("retail_test.gold_store_daily_sales").orderBy("date", "store_nbr").toPandas()
    )
    expected = dataset.train.groupby(["date", "store_nbr"]).sales.sum().reset_index()
    assert len(result) == len(expected)
    assert result.sales.tolist() == pytest.approx(expected.sales.tolist())
    transactions = spark.table("retail_test.gold_store_transactions").toPandas()
    assert transactions.units_per_transaction.tolist() == pytest.approx(
        (transactions.sales / 100).tolist()
    )


@pytest.mark.parametrize("defect", ["duplicate", "negative", "null", "header"])
def test_spark_csv_contract_rejects_bad_source(spark, dataset, tmp_path, defect):
    frame = dataset.train.iloc[:12].copy()
    if defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif defect == "negative":
        frame.loc[0, "sales"] = -1
    elif defect == "null":
        frame.loc[0, "family"] = None
    else:
        frame = frame.rename(columns={"sales": "wrong_column"})
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    with pytest.raises((ValueError, Py4JJavaError)):
        read_validated_csv(spark, str(path), "train")


def test_identifiers_prevent_sql_injection():
    assert namespace("workspace", "retail_forecasting") == "`workspace`.`retail_forecasting`"
    with pytest.raises(ValueError, match="Unsafe"):
        namespace("workspace", "x; DROP TABLE users")
