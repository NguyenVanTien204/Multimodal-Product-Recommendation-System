#!/usr/bin/env bash
# Creates the resource group + one VM with a daily auto-shutdown. Prints the cost before asking to proceed.
# Usage: deploy/azure/01-setup.sh [--yes]
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash on Windows would otherwise rewrite /subscriptions/... style arguments

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$here/config.env" ] || { echo "Missing $here/config.env (copy config.env.example)"; exit 1; }
# shellcheck disable=SC1091
source "$here/config.env"
command -v az >/dev/null || { echo "Azure CLI not found. Windows: winget install Microsoft.AzureCLI, then 'az login'."; exit 1; }
az account show >/dev/null 2>&1 || { echo "Not logged in: run 'az login' first."; exit 1; }
[ -f "$SSH_PUBLIC_KEY_FILE" ] || { echo "No SSH key at $SSH_PUBLIC_KEY_FILE. Create one: ssh-keygen -t ed25519"; exit 1; }

echo "Subscription: $(az account show --query '{name:name, id:id, state:state}' -o tsv)"
case "$(az account show --query name -o tsv)" in
  *Students*) ;;
  *) echo "WARNING: this does not look like an 'Azure for Students' subscription. Check 'az account list' before spending." ;;
esac

# --- pick a size that is really deployable here (student subscriptions block many SKUs) ----------------------
chosen=""
for size in $VM_SIZE $VM_FALLBACK_SIZES; do
  ok="$(az vm list-skus --location "$AZ_LOCATION" --size "$size" --resource-type virtualMachines \
        --query '[?length(restrictions)==`0`].name | [0]' -o tsv 2>/dev/null || true)"
  if [ -n "$ok" ]; then chosen="$size"; break; fi
  echo "  $size: not available to this subscription in $AZ_LOCATION"
done
[ -n "$chosen" ] || { echo "No candidate size is available in $AZ_LOCATION. Try another allowed region (AZ_LOCATION)."; exit 1; }

# --- cost preview (public retail prices, southeastasia, Linux; see README for the source) ----------------------
price="$(curl -s -G 'https://prices.azure.com/api/retail/prices' \
  --data-urlencode "\$filter=serviceName eq 'Virtual Machines' and armRegionName eq '$AZ_LOCATION' and armSkuName eq '$chosen' and priceType eq 'Consumption' and contains(productName,'Windows') eq false and contains(skuName,'Spot') eq false and contains(skuName,'Low Priority') eq false" \
  | python -c "import sys,json; i=json.load(sys.stdin).get('Items',[]); print(min(x['retailPrice'] for x in i) if i else 'unknown')" 2>/dev/null || echo unknown)"
echo
echo "About to create in '$AZ_LOCATION': $chosen (\$$price/hour while running), ${OS_DISK_GB} GB Standard SSD, 1 static public IP."
echo "Still billed while the VM is stopped (deallocated): disk + public IP, roughly \$8-9 per month."
echo "Auto-shutdown: ${AUTO_SHUTDOWN_UTC} UTC daily. Remember: only 'deallocate' stops compute billing."
if [ "${1:-}" != "--yes" ]; then read -r -p "Type 'yes' to continue: " reply; [ "$reply" = "yes" ] || { echo aborted; exit 1; }; fi

# --- resources -----------------------------------------------------------------------------------------------
az group create -n "$AZ_RG" -l "$AZ_LOCATION" --tags project=datn purpose=dev-demo -o none

my_ip="$(curl -s https://api.ipify.org)"
[ -n "$my_ip" ] || { echo "Could not detect your public IP for the SSH firewall rule"; exit 1; }

az vm create -g "$AZ_RG" -n "$VM_NAME" --image Ubuntu2204 --size "$chosen" \
  --os-disk-size-gb "$OS_DISK_GB" --storage-sku StandardSSD_LRS \
  --admin-username "$ADMIN_USER" --ssh-key-values "$SSH_PUBLIC_KEY_FILE" \
  --public-ip-sku Standard --nsg-rule NONE \
  --custom-data "$here/cloud-init.yaml" --tags project=datn -o table

# SSH only from this machine's current IP; nothing else is exposed (use `vm.sh tunnel` to reach the services).
az network nsg rule create -g "$AZ_RG" --nsg-name "${VM_NAME}NSG" -n ssh-from-my-ip --priority 1000 \
  --access Allow --protocol Tcp --direction Inbound --source-address-prefixes "$my_ip/32" \
  --destination-port-ranges 22 -o none

az vm auto-shutdown -g "$AZ_RG" -n "$VM_NAME" --time "$AUTO_SHUTDOWN_UTC" ${AUTO_SHUTDOWN_EMAIL:+--email "$AUTO_SHUTDOWN_EMAIL"} -o none

ip="$(az vm show -d -g "$AZ_RG" -n "$VM_NAME" --query publicIps -o tsv)"
echo
echo "VM ready: ssh $ADMIN_USER@$ip   (cloud-init needs ~3 min; wait for /var/lib/cloud/datn-ready)"
echo "SSH is allowed only from $my_ip. If your IP changes, update rule ssh-from-my-ip on ${VM_NAME}NSG."
echo "NEXT: create the portal budget (README, section 'Ngân sách') and check remaining credit."
