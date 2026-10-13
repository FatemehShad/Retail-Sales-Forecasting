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

import numpy as np
from pyspark.sql import functions as F

from retail_forecast.model import ForecastModel
from retail_forecast.pipeline import namespace

for name, default in [
    ("catalog", "workspace"),
    ("schema", "retail_forecasting"),
    ("model_path", ""),
    ("training_run_id", ""),
]:
    dbutils.widgets.text(name, default)
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
prefix = namespace(catalog, schema)
path = Path(dbutils.widgets.get("model_path"))
run_id = dbutils.widgets.get("training_run_id")
expected = Path(f"/Volumes/{catalog}/{schema}/models/{run_id}")
if not run_id or path != expected:
    raise ValueError("Only load the trusted model artifact produced by the upstream training task")
metadata = json.loads((path / "metadata.json").read_text())
model = ForecastModel.load(path)
history = spark.sql(
    f"SELECT * FROM {prefix}.silver_train VERSION AS OF {int(metadata['delta_versions']['train'])}"
).toPandas()
future = spark.sql(
    f"SELECT * FROM {prefix}.silver_test VERSION AS OF {int(metadata['delta_versions']['test'])}"
).toPandas()
forecasts = model.predict(history, future)
if (
    len(forecasts) != len(future)
    or set(forecasts.id) != set(future.id)
    or not np.isfinite(forecasts.sales).all()
    or (forecasts.sales < 0).any()
):
    raise RuntimeError("Invalid or incomplete forecast")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {prefix}.outputs")
output = Path(f"/Volumes/{catalog}/{schema}/outputs/{run_id}")
output.mkdir(parents=True, exist_ok=True)
forecasts[["id", "sales"]].to_csv(output / "submission.csv", index=False)
forecasts["training_run_id"] = run_id
forecasts["model"] = model.model_name
(
    spark.createDataFrame(forecasts)
    .withColumn("forecasted_at", F.current_timestamp())
    .write.format("delta")
    .mode("overwrite")
    .saveAsTable(f"{prefix}.gold_forecasts")
)
print(f"Validated {len(forecasts)} predictions; submission: {output / 'submission.csv'}")
