# Azure data solution

The supplied workspace is AWS-hosted. These steps apply to an Azure Databricks workspace and an Azure subscription with deployment permission; they are not executed against the supplied AWS workspace.

## ADLS Gen2 and Unity Catalog

`infra/azure/main.bicep` declares an ADLS Gen2 account and private container, an Azure Databricks Access Connector with a system-assigned identity, scoped Storage Blob Data Contributor access, and a Data Factory with managed identity. Account-key authentication and anonymous blob access are disabled. Configure network restrictions/private endpoints according to your Azure network and Databricks connectivity before production use.

```bash
az login
az deployment group create --resource-group <resource-group> \
  --template-file infra/azure/main.bicep \
  --parameters storageAccountName=<unique-lowercase-name> dataFactoryName=<unique-factory-name>
bash scripts/upload_adls.sh <storage-account> retail data/raw
```

The uploading Entra identity needs Storage Blob Data Contributor or equivalent file/ACL permissions. The upload script uses `--auth-mode login`, not a storage account key. No Azure subscription resources were provisioned or tested remotely during local development.

In **Azure Databricks**, replace the placeholders in `infra/azure/storage_credential.json` and `external_location.sql` using the Bicep outputs. An administrator creates the Unity Catalog storage credential using `databricks storage-credentials create --profile retail-azure --json @infra/azure/storage_credential.json`, then runs `external_location.sql`; the job principal needs `READ FILES` (and `WRITE FILES` if using external outputs). Run bootstrap for the managed model/output volumes and grant required schema/table/volume privileges. Then deploy with the Azure profile:

```bash
databricks auth login --host https://<azure-workspace>.azuredatabricks.net --profile retail-azure
databricks bundle validate -t azure --profile retail-azure \
  --var 'raw_data_path=abfss://retail@<storage-account>.dfs.core.windows.net/kaggle/store-sales'
databricks bundle deploy -t azure --profile retail-azure \
  --var 'raw_data_path=abfss://retail@<storage-account>.dfs.core.windows.net/kaggle/store-sales'
databricks bundle run -t azure retail_forecasting --profile retail-azure
```

The `azure` bundle target adds an Azure classic job cluster. Adjust node types, worker count, and runtime variables to available regional SKUs and workspace policy. Spark performs typed CSV ingestion/Delta transforms; driver-side Python performs rolling training and recursive scoring; SQL consumers read gold tables.

## Data Factory orchestration

`infra/azure/adf_pipeline.json` is an importable pipeline definition that calls the existing Databricks job with Data Factory's Entra managed identity, polls run status, and fails the pipeline if the Databricks result is not `SUCCESS`. It does not create a Databricks job, acquire Kaggle credentials, or copy competition files. Import it into the provisioned factory and supply the Azure workspace URL, workspace resource ID, and deployed job ID.

Add the factory's managed identity as a service principal in the Azure Databricks workspace and grant permission to run the job. Depending on workspace provisioning/authentication mode, Azure Contributor access to the workspace resource may also be needed; resolve through your workspace's managed identity configuration. The polling API requires read access to the run. Store no bearer tokens in this JSON. Execute the pipeline once manually and inspect the Databricks run and gold tables before configuring an ADF schedule or storage event trigger. An ADF `run-now` response only means the job was accepted; the polling and final result gate establish completion.

## Analytics consumers

The three gold historical aggregates and the current forecast table can be queried through Databricks SQL. Power BI can connect using a Databricks SQL warehouse with workspace-approved authentication. Actual transactions are used only for historical units-per-transaction analytics. Promotion associations are descriptive: the model and SQL do not claim causal marketing uplift.
