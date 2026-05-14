#!/usr/bin/env bash
set -euo pipefail

# Update UFW rules to allow only Cloudflare source IPs for 80/443.
# Requires root (sudo).

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root: sudo bash scripts/update_ufw_cloudflare.sh"
  exit 1
fi

CF_V4_URL="https://www.cloudflare.com/ips-v4"
CF_V6_URL="https://www.cloudflare.com/ips-v6"
APP_NAME_V4="Cloudflare-API-v4"
APP_NAME_V6="Cloudflare-API-v6"

tmp_v4="$(mktemp)"
tmp_v6="$(mktemp)"
trap 'rm -f "$tmp_v4" "$tmp_v6"' EXIT

echo "[1/6] Fetch Cloudflare IP lists..."
curl -fsSL "$CF_V4_URL" -o "$tmp_v4"
curl -fsSL "$CF_V6_URL" -o "$tmp_v6"

echo "[2/6] Ensure SSH is allowed..."
ufw allow OpenSSH >/dev/null || true
ufw allow 22/tcp >/dev/null || true

echo "[3/6] Remove old app-tagged Cloudflare rules (if any)..."
while read -r num _; do
  [[ -n "$num" ]] || continue
  ufw --force delete "$num" >/dev/null || true
done < <(ufw status numbered | sed -n "s/^\[\([0-9]\+\)\].*${APP_NAME_V4}.*/\1/p" | sort -rn)

while read -r num _; do
  [[ -n "$num" ]] || continue
  ufw --force delete "$num" >/dev/null || true
done < <(ufw status numbered | sed -n "s/^\[\([0-9]\+\)\].*${APP_NAME_V6}.*/\1/p" | sort -rn)

echo "[4/6] Add current Cloudflare IPv4 rules..."
while IFS= read -r cidr; do
  [[ -z "$cidr" ]] && continue
  ufw allow proto tcp from "$cidr" to any port 80 comment "$APP_NAME_V4" >/dev/null
  ufw allow proto tcp from "$cidr" to any port 443 comment "$APP_NAME_V4" >/dev/null
done < "$tmp_v4"

echo "[5/6] Add current Cloudflare IPv6 rules..."
while IFS= read -r cidr; do
  [[ -z "$cidr" ]] && continue
  ufw allow proto tcp from "$cidr" to any port 80 comment "$APP_NAME_V6" >/dev/null
  ufw allow proto tcp from "$cidr" to any port 443 comment "$APP_NAME_V6" >/dev/null
done < "$tmp_v6"

echo "[6/6] Ensure deny fallback for direct 80/443 access..."
ufw deny 80/tcp >/dev/null || true
ufw deny 443/tcp >/dev/null || true

echo
echo "Done. Current UFW status:"
ufw status
