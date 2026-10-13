# Retail Sales Forecasting

Forecast daily sales by store and product family over a 16-day horizon using the [Kaggle Store Sales dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).

## What it includes

- Python/LightGBM forecasting with historical lags, promotions, calendar, store, holiday, and oil features.
- Three rolling backtests using RMSLE, with a weekly seasonal baseline for model selection.
- Databricks ingestion, Delta tables, Spark SQL analytics, MLflow tracking, and Kaggle-format submissions.
- Azure ADLS Gen2 infrastructure and Data Factory orchestration templates.

## Quick start

Use Python 3.12 and Java 17 for local development. From the repository root:

```bash
bash scripts/setup.sh
source /workspace/.venvs/retail-forecasting/bin/activate
bash scripts/test.sh
```

Outside Codex, set `RETAIL_VENV_DIR="$PWD/.venv"` and `RETAIL_DELTA_DIR="$PWD/.local-tools/delta"` before setup, then activate `.venv`.

Accept Kaggle's competition rules and download these files into `data/raw`: `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, and `transactions.csv`.

```bash
retail-forecast train --data-dir data/raw --output-dir artifacts/run-01
```

The output includes model artifacts, backtest predictions, forecasts, and `submission.csv`. Keep data, credentials, and generated artifacts out of Git.

## Databricks and Azure

The default `dev` bundle targets the supplied AWS Databricks workspace using serverless notebooks. The separate `azure` target supports Azure Databricks and ADLS.

- [Databricks setup and execution](docs/databricks.md)
- [Azure infrastructure and pipelines](docs/azure.md)
- [Validation results and limitations](docs/validation.md)

**Status:** 29 local tests passed. Training on the actual Kaggle files and cloud execution remain unverified pending data and authenticated access. Synthetic test scores are not competition benchmarks.
