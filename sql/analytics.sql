-- Run after the job; adjust the catalog/schema to match bundle variables.
USE CATALOG workspace;
USE SCHEMA retail_forecasting;

-- Demand and promotion trends. This is descriptive, not causal promotion lift.
SELECT date, family, sales, promoted_items
FROM gold_family_daily_sales
ORDER BY date, family;

-- Store productivity uses historical transactions only.
SELECT date, store_nbr, city, sales, transactions, units_per_transaction
FROM gold_store_transactions
ORDER BY date DESC, sales DESC;

-- Compare each candidate across every rolling-origin fold.
SELECT training_run_id, model, COUNT(*) AS completed_folds, AVG(rmsle) AS mean_rmsle
FROM gold_backtest_metrics
GROUP BY training_run_id, model
ORDER BY training_run_id, mean_rmsle;

-- Current 16-day forecast, with its training lineage.
SELECT date, store_nbr, family, sales, model, training_run_id
FROM gold_forecasts
ORDER BY date, store_nbr, family;

-- Submission readiness: one non-negative, finite prediction per test id.
SELECT
  (SELECT COUNT(*) FROM silver_test) AS expected_rows,
  COUNT(*) AS forecast_rows,
  COUNT(DISTINCT id) AS unique_ids,
  SUM(CASE WHEN sales IS NULL OR ISNAN(sales) OR sales < 0 THEN 1 ELSE 0 END) AS invalid_rows
FROM gold_forecasts;
