#!/usr/bin/env bash
# Deletes EVERYTHING in the resource group (VM, disk, IP, NSG). Data on the VM is lost: snapshot it first.
set -euo pipefail
export MSYS_NO_PATHCONV=1
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$here/config.env"

az group show -n "$AZ_RG" --query '{name:name, location:location}' -o tsv >/dev/null || { echo "No resource group $AZ_RG"; exit 1; }
echo "Resources that will be deleted:"
az resource list -g "$AZ_RG" --query '[].{name:name, type:type}' -o table
read -r -p "Type the resource group name ($AZ_RG) to delete it: " reply
[ "$reply" = "$AZ_RG" ] || { echo aborted; exit 1; }
az group delete -n "$AZ_RG" --yes --no-wait
echo "Deletion started (runs in the background, a few minutes)."
