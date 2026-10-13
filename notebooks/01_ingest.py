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
from pyspark.sql import functions as F

from retail_forecast.pipeline import CSV_SCHEMAS, namespace, read_validated_csv, validate_panel

for name, default in [
    ("catalog", "workspace"),
    ("schema", "retail_forecasting"),
    ("raw_data_path", ""),
]:
    dbutils.widgets.text(name, default)
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
prefix = namespace(catalog, schema)
raw = dbutils.widgets.get("raw_data_path").rstrip("/") or f"/Volumes/{catalog}/{schema}/raw"
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {prefix}")
tables = {}
try:
    for name in CSV_SCHEMAS:
        tables[name] = read_validated_csv(spark, f"{raw}/{name}.csv", name)
    validate_panel(spark, tables)
    # Validate every source before replacing project-owned bronze snapshots.
    for name, frame in tables.items():
        (
            frame.withColumn("_source_file", F.lit(f"{raw}/{name}.csv"))
            .withColumn("_ingested_at", F.current_timestamp())
            .write.format("delta")
            .mode("overwrite")
            .saveAsTable(f"{prefix}.bronze_{name}")
        )
        print(f"bronze_{name}: {frame.count()} rows")
finally:
    tables.clear()
# Writes are atomic per Delta table. This six-table batch is not a cross-table transaction;
# dependent tasks start only on success. Do not run concurrent writers on this schema.
