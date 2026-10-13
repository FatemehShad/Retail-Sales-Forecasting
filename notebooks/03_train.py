# Databricks notebook source
from pathlib import Path

dbutils.widgets.text("project_path", str(Path.cwd().parent))
project_path = dbutils.widgets.get("project_path")
requirements_path = f"{project_path}/requirements-runtime.txt"

# COMMAND ----------
# MAGIC %pip install -r "$requirements_path"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import sys
from pathlib import Path

sys.path.insert(0, f"{dbutils.widgets.get('project_path')}/src")

# COMMAND ----------
import json
import os
from dataclasses import asdict

import mlflow
import pandas as pd

from retail_forecast.data import Dataset
from retail_forecast.model import ModelConfig, train_and_evaluate
from retail_forecast.pipeline import namespace

for name, default in [
    ("catalog", "workspace"),
    ("schema", "retail_forecasting"),
    ("experiment_path", "/Shared/retail-sales-forecasting"),
    ("folds", "3"),
    ("horizon", "16"),
    ("trees", "400"),
]:
    dbutils.widgets.text(name, default)
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
prefix = namespace(catalog, schema)
spark.sql(f"CREATE VOLUME IF NOT EXISTS {prefix}.models")
table_map = {
    "train": "silver_train",
    "test": "silver_test",
    "stores": "silver_stores",
    "oil": "silver_oil",
    "holidays_events": "silver_holidays",
    "transactions": "silver_transactions",
}
# Pin the Delta versions so repeated toPandas calls consume the same snapshots.
versions = {
    key: int(spark.sql(f"DESCRIBE HISTORY {prefix}.{table} LIMIT 1").first().version)
    for key, table in table_map.items()
}
frames = {
    key: spark.sql(f"SELECT * FROM {prefix}.{table} VERSION AS OF {versions[key]}").toPandas()
    for key, table in table_map.items()
}
data = Dataset.from_frames(**frames)
del frames
config = ModelConfig(
    folds=int(dbutils.widgets.get("folds")),
    horizon=int(dbutils.widgets.get("horizon")),
    n_estimators=int(dbutils.widgets.get("trees")),
    n_jobs=min(4, os.cpu_count() or 1),
)
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(dbutils.widgets.get("experiment_path"))
with mlflow.start_run(run_name="rolling-origin-forecast") as run:
    mlflow.log_params(asdict(config))
    mlflow.set_tags({"dataset": "store-sales-time-series-forecasting", "target": "sales"})
    model, metadata, backtests = train_and_evaluate(data, config)
    metadata["delta_versions"] = versions
    metadata["training_run_id"] = run.info.run_id
    metadata["catalog"] = catalog
    metadata["schema"] = schema
    metadata["tables"] = table_map
    metadata["package_version"] = "0.1.0"
    metadata["trained_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    mlflow.set_tag("selected_model", model.model_name)
    for name, score in metadata["mean_rmsle"].items():
        mlflow.log_metric(f"{name}_mean_rmsle", score)
    path = Path(f"/Volumes/{catalog}/{schema}/models/{run.info.run_id}")
    model.save(path, metadata)
    mlflow.log_artifact(str(path / "metadata.json"), "model")
    mlflow.log_artifact(str(path / "model.joblib"), "model")
    metrics = pd.DataFrame(metadata["folds"])
    metrics["training_run_id"] = run.info.run_id
    spark.createDataFrame(metrics).write.format("delta").mode("append").saveAsTable(
        f"{prefix}.gold_backtest_metrics"
    )
    backtests["training_run_id"] = run.info.run_id
    spark.createDataFrame(backtests).write.format("delta").mode("append").saveAsTable(
        f"{prefix}.gold_backtest_predictions"
    )
    print(json.dumps({"selected_model": model.model_name, "rmsle": metadata["mean_rmsle"]}))
    dbutils.jobs.taskValues.set(key="model_path", value=str(path))
    dbutils.jobs.taskValues.set(key="training_run_id", value=run.info.run_id)
