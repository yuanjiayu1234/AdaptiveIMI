#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_llama3_adaptiveimi.sh"

# This restart wrapper is intentionally Qwen3-only.  The underlying script can
# run llama/qwen/both, so force the family here even if the caller's shell has a
# different MODEL_FAMILY exported.
export MODEL_FAMILY=qwen3
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RESTART_SLEEP="${RESTART_SLEEP:-1}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"  # 0 means retry forever until success.

attempt=1
while true; do
  echo "[adaptiveimi-restart] $(date '+%F %T') attempt=${attempt} MODEL_FAMILY=${MODEL_FAMILY}" >&2

  "${RUN_SCRIPT}" "$@"
  status=$?

  if [[ "${status}" -eq 0 ]]; then
    echo "[adaptiveimi-restart] $(date '+%F %T') completed successfully" >&2
    exit 0
  fi

  # Preserve normal manual interruption behavior.
  if [[ "${status}" -eq 130 || "${status}" -eq 143 ]]; then
    echo "[adaptiveimi-restart] $(date '+%F %T') interrupted, exit=${status}" >&2
    exit "${status}"
  fi

  echo "[adaptiveimi-restart] $(date '+%F %T') failed, exit=${status}; restart after ${RESTART_SLEEP}s" >&2

  if [[ "${MAX_RESTARTS}" -gt 0 && "${attempt}" -ge "${MAX_RESTARTS}" ]]; then
    echo "[adaptiveimi-restart] reached MAX_RESTARTS=${MAX_RESTARTS}, stop" >&2
    exit "${status}"
  fi

  attempt=$((attempt + 1))
  sleep "${RESTART_SLEEP}"
done
