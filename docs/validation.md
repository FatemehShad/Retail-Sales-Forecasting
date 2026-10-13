# Validation record

Local development validation on 2026-10-13 used Python 3.12.14, NumPy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1, LightGBM 4.6.0, Spark 3.5.8, Delta Lake 3.3.2, and the host's Java 21.0.12.1. CI selects Java 17, the recommended local Spark runtime. Full local dependency versions are in `requirements.lock`.

## Executed checks

The 29 local tests cover:

- CSV round trips; duplicate keys/IDs; invalid targets; missing store metadata; non-consecutive or incomplete test panels.
- Target lags/rolling statistics excluding current and future sales, calendar gaps preserving the correct lag date, oil frozen at the forecast origin, no backward filling, holiday transfers and city/state scope.
- Known RMSLE calculations; real LightGBM fitting; recursive weekly baseline; model selection over six candidate/fold combinations; non-negative forecast coverage; artifact save/load equivalence; date objects returned by Spark; complete train/predict CLI execution.
- Real Spark CSV ingestion and Delta table writes/reads, SQL aggregate values and transaction joins, fail-fast headers/numeric/null/duplicate errors, safe SQL identifiers.
- Job task dependencies and model-path wiring; serverless dependency installation; Azure Data Factory managed-identity authentication, completion polling, and failure gating.

Spark/Delta tests use the local Delta V1 data-source writer through `spark.sql.sources.useV1SourceList`, with two shuffle/snapshot partitions. The default local Spark V2 overwrite route rejected Delta managed-table overwrites; the supported V1 configuration restored actual Delta writes/readback. This local configuration is separate from the Databricks runtime's native Delta integration. Spark 3.5 emits deprecation warnings about `distutils.version`; the tests do not suppress failures or alter assertions.

Additional checks: Ruff lint and formatting; `pip check`; shell syntax; Python parsing of exported notebooks; an actual wheel build; bundle configuration validation against the official Databricks CLI 0.279.0 JSON schema; Bicep 0.35.1 compilation of the Azure infrastructure template. Setup was rerun successfully in the current environment using `scripts/setup.sh`. These are local/static checks, not an Azure deployment or authenticated Databricks validation.

## Synthetic end-to-end demonstration

The seeded fixture (`make_dataset`, seed 42) contains two stores, three product families, 160 training days, known promotions, calendar effects, oil observations, and localized/transferred holidays. An independent CLI run used 120 trees, four model threads, three rolling 16-day folds, and the default minimum leaf size.

| Candidate | Mean rolling RMSLE on the synthetic fixture |
|---|---:|
| LightGBM | 0.046271 |
| Weekly seasonal baseline | 0.252585 |

LightGBM was selected. Scoring produced all 96 test IDs with finite, non-negative sales. Artifact reloading and a separate scoring CLI invocation are exercised by the automated tests. **These metrics describe generated test data, not the Kaggle competition or a production business dataset.**

## Outstanding cloud/data checks

The competition API returned HTTP 401 without Kaggle authentication. No competition CSVs were available for real-data training or accuracy measurement. An authenticated account must accept the rules and download the six CSV files, or provide an already downloaded governed storage folder.

The supplied workspace URL identifies AWS Databricks. `databricks bundle validate -t dev` could not configure authenticated credentials. The shell has no usable Databricks credential binding, and a browser/workspace URL alone does not grant API access. Thus job deployment, runtime compatibility, serverless memory limits, Unity Catalog privileges, MLflow cloud artifact uploads, and all four cloud task outcomes remain **unverified**. Follow `docs/databricks.md` after authenticating and uploading the files.

Azure Bicep compilation passed, but subscription permissions, resource deployment, ADLS managed-identity access, and live Data Factory polling were not exercised. The Azure templates are for an Azure workspace; the provided AWS workspace cannot use the Azure job-cluster configuration.

The rolling validation scores are used to select the candidate. Reserve an additional final holdout or obtain a competition evaluation before reporting unbiased model performance. No Kaggle submission was made.
