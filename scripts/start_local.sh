#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${ANIMA_HOST:-127.0.0.1}"
BASE_PORT="${ANIMA_PORT:-3000}"
CONFIG_SOURCE="${ROOT_DIR}/config.default.toml"
RUNNER="${ANIMA_RUNNER:-cargo}" # cargo|release
WAIT_SECS="${ANIMA_WAIT_SECS:-45}"
DETACHED=0
PRINT_ENV=0
ENV_FILE="${ANIMA_ENV_FILE:-${ROOT_DIR}/.anima-local.env}"

usage() {
  cat <<'EOF'
Usage: ./scripts/start_local.sh [--detached] [--print-env] [config.toml]

Options:
  --detached   Start server in background and exit when ready.
  --print-env  Print `export BASE_URL=...` when ready.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detached)
      DETACHED=1
      shift
      ;;
    --print-env)
      PRINT_ENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown flag: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      CONFIG_SOURCE="$1"
      shift
      ;;
  esac
done

say() {
  if [[ "${PRINT_ENV}" == "1" ]]; then
    echo "$*" >&2
  else
    echo "$*"
  fi
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

need_cmd curl
need_cmd python3
need_cmd cargo

if [[ ! -f "${CONFIG_SOURCE}" ]]; then
  echo "Config file not found: ${CONFIG_SOURCE}" >&2
  exit 1
fi

health_json() {
  local port="$1"
  curl -fsS "http://${HOST}:${port}/health" 2>/dev/null || true
}

pid_file_for_port() {
  local port="$1"
  echo "/tmp/anima-local.${port}.pid"
}

pidfile_running() {
  local port="$1"
  local pid_file pid
  pid_file="$(pid_file_for_port "${port}")"
  if [[ ! -f "${pid_file}" ]]; then
    return 1
  fi
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    return 1
  fi
  if kill -0 "${pid}" >/dev/null 2>&1; then
    return 0
  fi
  rm -f "${pid_file}"
  return 1
}

health_service() {
  local payload="$1"
  if [[ -z "${payload}" ]]; then
    echo ""
    return 0
  fi
  python3 - "${payload}" <<'PY'
import json, sys
try:
    data = json.loads(sys.argv[1])
except Exception:
    print("")
    raise SystemExit(0)
print(data.get("service", ""))
PY
}

port_in_use() {
  local port="$1"
  python3 - "${HOST}" "${port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.2)
try:
    rc = s.connect_ex((host, port))
finally:
    s.close()
raise SystemExit(0 if rc == 0 else 1)
PY
}

pick_port() {
  local port="${BASE_PORT}"
  local payload service
  local candidate

  payload="$(health_json "${port}")"
  service="$(health_service "${payload}")"
  if [[ "${service}" == "anima" ]]; then
    echo "${port}"
    return 0
  fi

  if port_in_use "${port}"; then
    # Prefer already-running anima instances before selecting a new free port.
    for candidate in 3010 3011 3012 3013 3014 3015; do
      payload="$(health_json "${candidate}")"
      service="$(health_service "${payload}")"
      if [[ "${service}" == "anima" ]]; then
        echo "${candidate}"
        return 0
      fi
    done

    # Reuse port where anima startup is already in progress.
    for candidate in 3010 3011 3012 3013 3014 3015; do
      if pidfile_running "${candidate}"; then
        echo "${candidate}"
        return 0
      fi
    done

    for candidate in 3010 3011 3012 3013 3014 3015; do
      if ! port_in_use "${candidate}"; then
        echo "${candidate}"
        return 0
      fi
    done
    echo "Could not find a free port in fallback range 3010-3015" >&2
    exit 1
  fi

  echo "${port}"
}

SELECTED_PORT="$(pick_port)"
EXISTING_PAYLOAD="$(health_json "${SELECTED_PORT}")"
EXISTING_SERVICE="$(health_service "${EXISTING_PAYLOAD}")"

if [[ "${EXISTING_SERVICE}" == "anima" ]]; then
  BASE_URL="http://${HOST}:${SELECTED_PORT}"
  if [[ "${PRINT_ENV}" == "1" ]]; then
    echo "export BASE_URL=${BASE_URL}"
  else
    say "Anima is already running at ${BASE_URL}"
    say "Health: ${EXISTING_PAYLOAD}"
  fi
  exit 0
fi

if pidfile_running "${SELECTED_PORT}"; then
  BASE_URL="http://${HOST}:${SELECTED_PORT}"
  if [[ "${PRINT_ENV}" == "1" ]]; then
    echo "export BASE_URL=${BASE_URL}"
  else
    say "Anima startup already in progress: ${BASE_URL}"
    say "Wait a few seconds then check: curl -sS ${BASE_URL}/health"
  fi
  exit 0
fi

if [[ "${SELECTED_PORT}" != "${BASE_PORT}" ]]; then
  say "Port ${BASE_PORT} is occupied by a non-anima service; starting anima on ${SELECTED_PORT}."
fi

TMP_CONFIG="$(mktemp "/tmp/anima-local.XXXXXX")"
cleanup() {
  rm -f "${TMP_CONFIG}"
}
trap cleanup EXIT

cp "${CONFIG_SOURCE}" "${TMP_CONFIG}"
ANIMA_SELECTED_PORT="${SELECTED_PORT}" perl -0777 -i -pe \
  's/(\[server\]\s*host\s*=\s*".*?"\s*port\s*=\s*)\d+/${1}$ENV{ANIMA_SELECTED_PORT}/s' \
  "${TMP_CONFIG}"
perl -0777 -i -pe 's/(\[consolidation\]\s*enabled\s*=\s*)true/${1}false/s' "${TMP_CONFIG}"
perl -0777 -i -pe 's/(\[processor\]\s*enabled\s*=\s*)true/${1}false/s' "${TMP_CONFIG}"

if [[ "${RUNNER}" == "release" ]]; then
  SERVER_CMD=("${ROOT_DIR}/target/release/anima-server" "${TMP_CONFIG}")
else
  SERVER_CMD=(cargo run -p anima-server --bin anima-server -- "${TMP_CONFIG}")
fi

say "Starting anima on http://${HOST}:${SELECTED_PORT}"
say "Config: ${TMP_CONFIG} (consolidation/processor disabled for local smoke)"

if [[ "${DETACHED}" == "1" ]]; then
  LOG_FILE="/tmp/anima-local.${SELECTED_PORT}.log"
  pushd "${ROOT_DIR}" >/dev/null
  nohup "${SERVER_CMD[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
  SERVER_PID=$!
  disown "${SERVER_PID}" 2>/dev/null || true
  popd >/dev/null
  printf '%s\n' "${SERVER_PID}" > "$(pid_file_for_port "${SELECTED_PORT}")"
  say "Logs: ${LOG_FILE}"
else
  (
    cd "${ROOT_DIR}"
    "${SERVER_CMD[@]}"
  ) &
  SERVER_PID=$!
fi

if [[ "${DETACHED}" == "1" ]]; then
  BASE_URL="http://${HOST}:${SELECTED_PORT}"
  printf 'export BASE_URL=%s\n' "${BASE_URL}" > "${ENV_FILE}"
  if [[ "${PRINT_ENV}" == "1" ]]; then
    echo "export BASE_URL=${BASE_URL}"
  else
    say "Anima is starting in background: ${BASE_URL}"
    say "Env file: ${ENV_FILE}"
    say "Run: source ${ENV_FILE}"
    say "Tip: wait a few seconds, then curl ${BASE_URL}/health"
  fi
  exit 0
fi

for _ in $(seq 1 "${WAIT_SECS}"); do
  payload="$(health_json "${SELECTED_PORT}")"
  service="$(health_service "${payload}")"
  if [[ "${service}" == "anima" ]]; then
    BASE_URL="http://${HOST}:${SELECTED_PORT}"
    say "Anima is ready: ${BASE_URL}"
    say "Example:"
    say "  curl -sS ${BASE_URL}/health"
    wait "${SERVER_PID}"
    exit $?
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    wait "${SERVER_PID}" || true
    echo "Server process exited before becoming ready." >&2
    exit 1
  fi
  sleep 1
done

echo "Timed out waiting for anima health endpoint." >&2
kill "${SERVER_PID}" >/dev/null 2>&1 || true
wait "${SERVER_PID}" || true
exit 1
