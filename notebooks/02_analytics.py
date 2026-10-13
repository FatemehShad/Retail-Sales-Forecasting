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
from retail_forecast.pipeline import analytics_queries, namespace, silver_queries

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "retail_forecasting")
prefix = namespace(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
for name, query in silver_queries(prefix).items():
    spark.sql(f"CREATE OR REPLACE TABLE {prefix}.{name} USING DELTA AS {query}")
for name, query in analytics_queries(prefix).items():
    spark.sql(f"CREATE OR REPLACE TABLE {prefix}.{name} USING DELTA AS {query}")
    print(name, spark.table(f"{prefix}.{name}").count())
