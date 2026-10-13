-- Azure Databricks only. Replace placeholders using the Bicep outputs before running.
-- First create retail_adls_identity via the CLI using storage_credential.json.
-- Requires metastore administration / CREATE STORAGE CREDENTIAL and EXTERNAL LOCATION.

CREATE EXTERNAL LOCATION IF NOT EXISTS retail_adls
URL 'abfss://retail@<storage-account>.dfs.core.windows.net/'
WITH (STORAGE CREDENTIAL retail_adls_identity);

GRANT READ FILES, WRITE FILES ON EXTERNAL LOCATION retail_adls TO `<job-principal>`;

-- Optionally expose the raw folder as a UC external volume.
CREATE EXTERNAL VOLUME IF NOT EXISTS workspace.retail_forecasting.adls_raw
LOCATION 'abfss://retail@<storage-account>.dfs.core.windows.net/kaggle/store-sales';
