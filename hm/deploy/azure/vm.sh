#!/usr/bin/env bash
# Day-to-day control of the DATN VM.  Usage: hm/deploy/azure/vm.sh start|stop|status|ssh|tunnel
set -euo pipefail
export MSYS_NO_PATHCONV=1
command -v az >/dev/null || export PATH="$PATH:/c/Program Files/Microsoft SDKs/Azure/CLI2/wbin"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$here/config.env"

power() { az vm get-instance-view -g "$AZ_RG" -n "$VM_NAME" --query "instanceView.statuses[?starts_with(code,'PowerState/')].displayStatus | [0]" -o tsv; }
ip()    { az vm show -d -g "$AZ_RG" -n "$VM_NAME" --query publicIps -o tsv; }

case "${1:-}" in
  start)  az vm start -g "$AZ_RG" -n "$VM_NAME" -o none; echo "started: $(power)  ip=$(ip)" ;;
  # `az vm stop` keeps the VM allocated and STILL bills compute. Always deallocate.
  stop)   az vm deallocate -g "$AZ_RG" -n "$VM_NAME" -o none; echo "now: $(power)  (disk + IP still billed, compute is not)" ;;
  status) echo "$(power)  ip=$(ip)" ;;
  ssh)    exec ssh "$ADMIN_USER@$(ip)" ;;
  # Forward the services to localhost so nothing has to be exposed publicly: 3000 web, 8000 backend, 8100 recommender, 8200 rag, 6333 qdrant.
  tunnel) exec ssh -N -L 3000:localhost:3000 -L 8000:localhost:8000 -L 8100:localhost:8100 -L 8200:localhost:8200 -L 6333:localhost:6333 "$ADMIN_USER@$(ip)" ;;
  ps)     ssh "$ADMIN_USER@$(ip)" 'cd ~/datn && docker compose --profile ai ps -a; free -h | head -2' ;;
  logs)   exec ssh "$ADMIN_USER@$(ip)" "cd ~/datn && docker compose --profile ai logs -f --tail=100 ${2:-}" ;;
  *) echo "usage: $0 start|stop|status|ssh|tunnel|logs [service]|ps"; exit 1 ;;
esac
