#!/usr/bin/env bash
# Uses the Azure CLI's signed-in Entra identity; no storage account key required.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: bash scripts/upload_adls.sh STORAGE_ACCOUNT FILESYSTEM LOCAL_CSV_DIRECTORY" >&2
  exit 2
fi
RETAIL_ACCOUNT="$1"
RETAIL_FILESYSTEM="$2"
RETAIL_DATA_DIR="$3"
for RETAIL_TABLE in train test stores oil holidays_events transactions; do
  test -f "$RETAIL_DATA_DIR/$RETAIL_TABLE.csv"
done
az storage fs directory create --account-name "$RETAIL_ACCOUNT" \
  --file-system "$RETAIL_FILESYSTEM" --name kaggle --auth-mode login --output none
az storage fs directory create --account-name "$RETAIL_ACCOUNT" \
  --file-system "$RETAIL_FILESYSTEM" --name kaggle/store-sales --auth-mode login --output none
for RETAIL_TABLE in train test stores oil holidays_events transactions; do
  az storage fs file upload --account-name "$RETAIL_ACCOUNT" \
    --file-system "$RETAIL_FILESYSTEM" --path "kaggle/store-sales/$RETAIL_TABLE.csv" \
    --source "$RETAIL_DATA_DIR/$RETAIL_TABLE.csv" --auth-mode login --overwrite true --output none
done
