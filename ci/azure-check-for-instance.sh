#!/usr/bin/env bash

set -euo pipefail

# Checks if an instance of a given name and size exists in the Azure resource group

declare -g AZURE_RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:?}"
declare -g AZURE_INSTANCE_SIZE="${AZURE_INSTANCE_SIZE:?}"
declare -g AZURE_INSTANCE_NAME="${AZURE_INSTANCE_NAME:?}"

az vm show \
  --show-details \
  --resource-group "$AZURE_INSTANCE_RESOURCE_GROUP" \
  --name "$AZURE_INSTANCE_NAME"
