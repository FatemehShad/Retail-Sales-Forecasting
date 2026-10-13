# Databricks setup and operation

## Workspace and access

The provided `dbc-9b6ad1bb-15c7.cloud.databricks.com` address is an AWS workspace, not Azure Databricks. `dev` uses it explicitly and defaults to serverless notebook jobs. `azure` uses `DATABRICKS_HOST` or a selected CLI profile and adds Azure job compute. `prod` is an unscheduled serverless target; configure its service-principal run identity and job permissions for production use.

Required access: permission to create a job and a workspace experiment, use serverless or approved classic compute, `USE CATALOG`, and permission to create/use a dedicated schema, managed volumes, and Delta tables. The catalog must have managed storage. `workspace` is a convenient default, not a guarantee that the catalog exists or that its permissions allow this workflow. Override `catalog` and `schema` to project-owned locations.

Install the [official Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) and authenticate with OAuth using `databricks auth login --host <workspace-url>`. The project configuration was checked against CLI 0.279.0's schema locally. Use an authenticated `bundle validate` to check actual workspace capabilities. No workspace deployment was validated without credentials.

For automation, use workspace-approved OAuth or a securely injected `DATABRICKS_TOKEN` and non-secret `DATABRICKS_HOST`. Do not commit tokens. Codex proxy-injected credentials apply only to their configured HTTPS destinations. An authenticated browser session does not automatically authenticate the development shell.

## Bootstrap and upload

1. Add this repository as a Databricks Git folder, or deploy the bundle from an authenticated local checkout.
2. Run `notebooks/00_bootstrap.py` using a serverless environment with Python >=3.11, or Databricks Runtime 16.4 LTS. For an imported Git folder, the default `project_path` is the notebook's parent folder; override it if your workspace uses a different current directory. For a bundle-deployed notebook, set `project_path` to the bundle's synchronized `files` folder. Set `catalog`/`schema` to your chosen namespace.
3. Upload the extracted six CSV files to `/Volumes/<catalog>/<schema>/raw`. The bootstrap creates `raw`, `models`, and `outputs` managed volumes. For the Azure external-storage option, use the [Azure instructions](azure.md).
4. Validate, deploy, and run the job:

```bash
databricks bundle validate -t dev --var catalog=workspace,schema=retail_forecasting
databricks bundle deploy -t dev --var catalog=workspace,schema=retail_forecasting
databricks bundle run -t dev retail_forecasting
```

The `raw_data_path` variable defaults to the raw volume. For another governed folder, override it at validation, deployment, and run consistently, or configure a bundle variable override file excluded from Git. `abfss://...` is available on Azure only with Unity Catalog external-location permission. Cross-cloud storage is not assumed.

The source notebooks install `requirements-runtime.txt` with `%pip`, restart Python, then load `src/retail_forecast` via the `project_path` job parameter. Serverless tasks have no cluster-bound `libraries`; installing dependencies in notebook cells works for notebook tasks. Package downloads require approved PyPI egress. The wheel artifact is provided for reuse on supported clusters and for packaging verification.

## Outputs and required cloud validation

On a successful run, confirm all four tasks complete: `ingest`, `analytics`, `train`, `forecast`. Then run `sql/analytics.sql` in a SQL warehouse or notebook, using your catalog/schema.

- Bronze sources and silver tables must have expected row counts, with the test horizon starting one day after the final train date.
- `gold_backtest_metrics` must contain three completed folds for **each** candidate for the training run. Check that the selected candidate has the lower mean RMSLE or that the weekly baseline is retained on a tie.
- `gold_backtest_predictions` contains actuals and recursive forecasts for diagnostics; calculate segmented RMSLE by family/store to identify weaker series.
- `gold_forecasts` must cover exactly the IDs in `silver_test`, with finite, non-negative sales.
- MLflow must show parameters, both mean RMSLE metrics, selected-model tag, and artifacts for the run. Models/metadata are additionally stored in `/Volumes/<catalog>/<schema>/models/<run-id>`.
- Download `/Volumes/<catalog>/<schema>/outputs/<run-id>/submission.csv` and confirm columns are exactly `id,sales`. Competition submission itself is a separate user action.

The model runs on the driver via pandas/LightGBM. Three rolling folds plus final fitting may need substantial driver memory. The Azure target starts with a 32 GB `Standard_D8ds_v5` driver, two `Standard_D4ds_v5` workers, and four model threads. Serverless sizing depends on workspace capability. Change compute through bundle variables/approved policies; do not confuse installed dependencies with capacity verification.

## Repeatability and lineage

The job rebuilds project-owned bronze/silver/analytics snapshots, appends uniquely identified training metrics/backtests, and replaces the current gold forecast snapshot. Each model/submission has a unique MLflow run-ID directory. Do not point the job at shared tables or run concurrent writers on this schema. Delta commits are atomic per table; the CSV batch does not implement a transaction spanning six tables. A failed ingest must be repaired and rerun before downstream consumption.

Training pins all six silver table versions. Scoring uses the recorded train/test versions, preventing a later ingestion from changing the forecast inputs. Keep Delta retention long enough for model runs that need rescoring; vacuumed versions cannot be reconstructed by metadata alone. The pipeline is an explicit batch workflow for this static competition dataset, not a streaming change-data-capture service.

If running notebooks manually, run them in numeric order. `03_train`/`04_forecast` are job tasks: the training task publishes `model_path` and `training_run_id` through `dbutils.jobs.taskValues`. For a manual scoring notebook, set these widgets to the trusted upstream run's artifact path and ID; all data must already have been validated by ingest.
