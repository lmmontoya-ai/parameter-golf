#!/usr/bin/env bash
set -euo pipefail

mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh

if command -v ssh-keygen >/dev/null 2>&1; then
  ssh-keygen -A >/dev/null 2>&1 || true
fi

if [[ -n "${AUTHORIZED_KEYS:-}" ]]; then
  printf '%s\n' "${AUTHORIZED_KEYS}" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

if [[ "${START_SSHD:-1}" = "1" ]]; then
  /usr/sbin/sshd
fi

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

exec sleep infinity
