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
# Run once, then upload the six CSVs into the raw volume before running the job.
from retail_forecast.pipeline import namespace

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "retail_forecasting")
prefix = namespace(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {prefix}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {prefix}.raw")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {prefix}.models")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {prefix}.outputs")
print("Volumes created. Upload train/test/stores/oil/holidays_events/transactions.csv into raw.")
