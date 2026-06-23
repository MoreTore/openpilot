#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Run tools/install_python_dependencies.sh first."
  exit 1
fi

default_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  else
    echo 8
  fi
}

env_var_truthy() {
  [[ "${1:-}" =~ ^(1|true|yes|on)$ ]]
}

usage() {
  cat <<'EOF'
Usage:
  ./onroad [jobs] [--c3 | --c4 | --all | --replay-only] [--galaxy] [-nav] [--offroad] [-alert] [--cem] [--prefix name] <route-or-replay-args...>

Examples:
  ./onroad <route>
  ./onroad --c3 <route>
  ./onroad --c4 <route> --start 30
  ./onroad --c4 -nav <route>
  ./onroad --c3 --nav --offroad --demo
  ./onroad --c4 --cem --demo
  ./onroad --c3 --cem --demo
  ./onroad --c3 --cem -alert --demo --no-loop
  ./onroad --all <route>
  ./onroad --replay-only --demo --no-vipc --no-loop

Notes:
  - This is host/dev only. It uses the isolated host worktree and does not touch the device path.
  - A private comma connect route still requires tools/lib/auth.py before replay can download it.
  - If no UI flag is provided, the route's logged device type selects the UI: mici/c4 -> c4, all big devices -> c3.
  - Use multiple UI flags together if you want more than one desktop UI at once.
  - --galaxy starts a local Galaxy web session with the same preview params and prints the localhost URL. It blocks replay's logged customReserved9 stream so Galaxy can own the live Testing Grounds publisher.
  - -nav injects a fake navigation demo stream and blocks replay from publishing navInstruction/navRoute.
  - --offroad is only valid with --nav; together they preview the offroad Quick Start card and do not start the fake on-road nav publisher.
  - --cem publishes fake CEM statuses for desktop visual review in the raylib UIs.
  - --csc publishes a fake starpilotPlan stream that forces the CSC glow to render on desktop UI.
  - -alert blocks replay from publishing selfdriveState and fires a fake critical full-screen red alert (alertSize=full, alertStatus=critical) 20 seconds after the demo publisher starts (10s for replay route + UI to come up, plus 10s for the user to open Settings). Default alert text mimics a real controlsMismatch event; run tools/replay/fake_alert_demo.py directly to override --text1/--text2/--delay.
EOF
}

jobs="$(default_jobs)"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  jobs="$1"
  shift || true
fi

PREFIX_ARG=""
REPLAY_ARGS=()
UI_TARGETS=()
UI_SELECTION_EXPLICIT=0
LEGACY_UI_SELECTION=""
REPLAY_ONLY=0
NAV_DEMO=0
OFFROAD_DEMO=0
CEM_DEMO=0
ALERT_DEMO=0
CSC_DEMO=0
GALAXY=0
LIVE_CONTROLS=0
REPLAY_PID=""
NAV_PID=""
CEM_PID=""
ALERT_PID=""
CSC_PID=""
GALAXY_PID=""
GPU_SYNC_PID=""
GALAXY_PORT=""
GALAXY_URL=""
ONROAD_TEMP_PREFIX=""
UI_PIDS=()
LIVE_PIDS=()

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --c3)
        UI_TARGETS+=(c3)
        UI_SELECTION_EXPLICIT=1
        shift
        ;;
      --c4)
        UI_TARGETS+=(c4)
        UI_SELECTION_EXPLICIT=1
        shift
        ;;
      --all)
        UI_TARGETS+=(c3 c4)
        UI_SELECTION_EXPLICIT=1
        shift
        ;;
      --replay-only)
        REPLAY_ONLY=1
        shift
        ;;
      -nav|--nav)
        NAV_DEMO=1
        shift
        ;;
      --offroad)
        OFFROAD_DEMO=1
        shift
        ;;
      --cem|--mici-widget-demo|--widget-demo)
        CEM_DEMO=1
        shift
        ;;
      --csc|--csc-demo)
        CSC_DEMO=1
        shift
        ;;
      --galaxy)
        GALAXY=1
        shift
        ;;
      -alert|--alert|--alert-demo)
        ALERT_DEMO=1
        shift
        ;;
      --live|--live-controls)
        LIVE_CONTROLS=1
        shift
        ;;
      --ui)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --ui" >&2
          exit 1
        fi
        LEGACY_UI_SELECTION="$2"
        UI_SELECTION_EXPLICIT=1
        shift 2
        ;;
      --ui=*)
        LEGACY_UI_SELECTION="${1#*=}"
        UI_SELECTION_EXPLICIT=1
        shift
        ;;
      -p|--prefix)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for $1" >&2
          exit 1
        fi
        PREFIX_ARG="$2"
        shift 2
        ;;
      --prefix=*)
        PREFIX_ARG="${1#*=}"
        shift
        ;;
      --)
        shift
        REPLAY_ARGS+=("$@")
        break
        ;;
      *)
        REPLAY_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

expand_ui_targets() {
  local selection="${1// /}"

  case "${selection,,}" in
    all|"")
      UI_TARGETS=(c3 c4)
      return
      ;;
    none)
      UI_TARGETS=()
      return
      ;;
  esac

  local raw_targets=()
  IFS=',' read -r -a raw_targets <<< "${selection}"

  local normalized=()
  local raw=""
  for raw in "${raw_targets[@]}"; do
    case "${raw,,}" in
      c3|c4)
        normalized+=("${raw,,}")
        ;;
      *)
        echo "Unknown UI target in --ui: ${raw}" >&2
        echo "Valid values: all, none, c3, c4" >&2
        exit 1
        ;;
    esac
  done

  local ordered_targets=(c3 c4)
  local target=""
  for target in "${ordered_targets[@]}"; do
    local candidate=""
    for candidate in "${normalized[@]}"; do
      if [[ "${candidate}" == "${target}" ]]; then
        UI_TARGETS+=("${target}")
        break
      fi
    done
  done
}

dedupe_ui_targets() {
  local ordered_targets=(c3 c4)
  local deduped=()
  local target=""
  for target in "${ordered_targets[@]}"; do
    local candidate=""
    for candidate in "${UI_TARGETS[@]-}"; do
      if [[ "${candidate}" == "${target}" ]]; then
        deduped+=("${target}")
        break
      fi
    done
  done
  UI_TARGETS=("${deduped[@]-}")
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  local pid=""
  for pid in "${UI_PIDS[@]-}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  if [[ -n "${REPLAY_PID}" ]]; then
    kill "${REPLAY_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${NAV_PID}" ]]; then
    kill "${NAV_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${CEM_PID}" ]]; then
    kill "${CEM_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${ALERT_PID}" ]]; then
    kill "${ALERT_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${CSC_PID}" ]]; then
    kill "${CSC_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GALAXY_PID}" ]]; then
    kill "${GALAXY_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GPU_SYNC_PID}" ]]; then
    kill "${GPU_SYNC_PID}" >/dev/null 2>&1 || true
  fi
  for pid in "${LIVE_PIDS[@]-}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done

  for pid in "${UI_PIDS[@]-}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" >/dev/null 2>&1 || true
    fi
  done
  if [[ -n "${REPLAY_PID}" ]]; then
    wait "${REPLAY_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${NAV_PID}" ]]; then
    wait "${NAV_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${CEM_PID}" ]]; then
    wait "${CEM_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${ALERT_PID}" ]]; then
    wait "${ALERT_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${CSC_PID}" ]]; then
    wait "${CSC_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GALAXY_PID}" ]]; then
    wait "${GALAXY_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GPU_SYNC_PID}" ]]; then
    wait "${GPU_SYNC_PID}" >/dev/null 2>&1 || true
  fi
  for pid in "${LIVE_PIDS[@]-}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" >/dev/null 2>&1 || true
    fi
  done

  if [[ -n "${ONROAD_TEMP_PREFIX:-}" && "${ONROAD_TEMP_PREFIX}" == desktop-onroad-* ]]; then
    echo "Cleaning up temporary prefix environment (${ONROAD_TEMP_PREFIX})..."
    rm -rf "/dev/shm/msgq_${ONROAD_TEMP_PREFIX}"
    rm -rf "/tmp/comma_download_cache${ONROAD_TEMP_PREFIX}"
    rm -rf "${HOME}/.comma${ONROAD_TEMP_PREFIX}"
  fi

  exit "${exit_code}"
}

append_blocked_service_names() {
  local existing="$1"
  local required="$2"
  local combined="${existing}"

  if [[ -n "${combined}" ]]; then
    combined+=",${required}"
  else
    combined="${required}"
  fi

  local deduped=()
  local seen=","
  local part=""
  local parts=()
  IFS=',' read -r -a parts <<< "${combined}"
  for part in "${parts[@]-}"; do
    part="${part// /}"
    if [[ -z "${part}" ]]; then
      continue
    fi
    if [[ "${seen}" != *",${part},"* ]]; then
      deduped+=("${part}")
      seen+="${part},"
    fi
  done

  local joined=""
  for part in "${deduped[@]-}"; do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${part}"
  done

  printf '%s' "${joined}"
}

ensure_replay_blocklist() {
  local services="$1"
  local idx=0

  for ((idx=0; idx<${#REPLAY_ARGS[@]}; idx++)); do
    case "${REPLAY_ARGS[$idx]}" in
      -b|--block)
        if (( idx + 1 >= ${#REPLAY_ARGS[@]} )); then
          echo "Missing value for ${REPLAY_ARGS[$idx]}" >&2
          exit 1
        fi
        REPLAY_ARGS[$((idx + 1))]="$(append_blocked_service_names "${REPLAY_ARGS[$((idx + 1))]}" "${services}")"
        return
        ;;
      --block=*)
        REPLAY_ARGS[$idx]="--block=$(append_blocked_service_names "${REPLAY_ARGS[$idx]#*=}" "${services}")"
        return
        ;;
    esac
  done

  REPLAY_ARGS=(-b "${services}" "${REPLAY_ARGS[@]}")
}

ensure_nav_demo_replay_blocklist() {
  ensure_replay_blocklist "navInstruction,navRoute"
}

ensure_alert_demo_replay_blocklist() {
  ensure_replay_blocklist "selfdriveState"
}

ensure_csc_demo_replay_blocklist() {
  ensure_replay_blocklist "starpilotPlan"
}

ensure_galaxy_replay_blocklist() {
  ensure_replay_blocklist "customReserved9"
}

ensure_live_controls_replay_blocklist() {
  local blocked_services="carState,carParams,carControl,controlsState,sendcan,carOutput"
  local idx=0

  for ((idx=0; idx<${#REPLAY_ARGS[@]}; idx++)); do
    case "${REPLAY_ARGS[$idx]}" in
      -b|--block)
        if (( idx + 1 >= ${#REPLAY_ARGS[@]} )); then
          echo "Missing value for ${REPLAY_ARGS[$idx]}" >&2
          exit 1
        fi
        REPLAY_ARGS[$((idx + 1))]="$(append_blocked_service_names "${REPLAY_ARGS[$((idx + 1))]}" "${blocked_services}")"
        return
        ;;
      --block=*)
        REPLAY_ARGS[$idx]="--block=$(append_blocked_service_names "${REPLAY_ARGS[$idx]#*=}" "${blocked_services}")"
        return
        ;;
    esac
  done

  REPLAY_ARGS=(-b "${blocked_services}" "${REPLAY_ARGS[@]}")
}

prepare_env() {
  source .venv/bin/activate

  if [[ -d /opt/homebrew/bin ]]; then
    export PATH="/opt/homebrew/bin:${PATH}"
  fi

  export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/starpilot/third_party"
  local repo_dir=""
  for repo_dir in "${ROOT_DIR}"/*_repo; do
    [[ -d "${repo_dir}" ]] && export PYTHONPATH="${PYTHONPATH}:${repo_dir}"
  done
  [[ -d "${ROOT_DIR}/third_party/acados" ]] && export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/third_party/acados"

  export BASEDIR="${ROOT_DIR}"
  export NOBOARD=1
  export SIMULATION=1
  export SKIP_FW_QUERY=1
  export USE_WEBCAM=1
  export SP_C3_FAKE_WIFI=0
  export SP_C4_FAKE_WIFI=0
  export SP_ALLOW_DESKTOP_FAKE_WIFI=0
  export SP_ONROAD_NAV_DEMO="${NAV_DEMO}"
  export SP_ONROAD_OFFROAD_DEMO="${OFFROAD_DEMO}"
  export SP_CEM_DEMO="${CEM_DEMO}"
  export SP_ONROAD_ALERT_DEMO="${ALERT_DEMO}"
  export SP_ONROAD_CSC_DEMO="${CSC_DEMO}"

  local generated_prefix="${PREFIX_ARG:-${OPENPILOT_PREFIX:-desktop-onroad-$$}}"
  ONROAD_TEMP_PREFIX="${generated_prefix}"

  if [[ "$(uname -s)" == "Darwin" ]] || env_var_truthy "${ZMQ:-0}"; then
    export OPENPILOT_ZMQ_NAMESPACE="${PREFIX_ARG:-${OPENPILOT_ZMQ_NAMESPACE:-${generated_prefix}}}"
    unset OPENPILOT_PREFIX
    export PARAMS_ROOT="${PARAMS_ROOT:-${HOME}/.comma${generated_prefix}/params}"
  else
    export OPENPILOT_PREFIX="${generated_prefix}"
    mkdir -p "/dev/shm/msgq_${OPENPILOT_PREFIX}"
  fi
}

seed_params() {
  "${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/tools/replay/onroad_config.py" seed "${REPLAY_ARGS[@]}"
}

auto_select_ui_targets() {
  local selection=""
  selection="$("${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/tools/replay/onroad_config.py" select-ui "${REPLAY_ARGS[@]}")"
  IFS=' ' read -r -a UI_TARGETS <<< "${selection}"
  dedupe_ui_targets
  echo "Auto-selected UI: ${UI_TARGETS[*]}"
}

seed_starpilot_theme() {
  "${ROOT_DIR}/.venv/bin/python3" - <<'PY'
from openpilot.starpilot.common.starpilot_functions import seed_desktop_theme_assets

seed_desktop_theme_assets()
PY
}

build_replay() {
  local targets=(
    tools/replay/replay
    common/params_pyx.so
    common/transformations/transformations.so
    msgq_repo/msgq/ipc_pyx.so
    msgq_repo/msgq/visionipc/visionipc_pyx.so
  )
  if [[ "${LIVE_CONTROLS}" == "1" ]]; then
    targets+=(
      selfdrive/pandad/pandad_api_impl.so
      selfdrive/modeld/models/commonmodel_pyx.so
      selfdrive/controls/lib/lateral_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so
      selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so
    )
  fi
  "${ROOT_DIR}/.venv/bin/scons" --cache-disable --extras -j"${jobs}" "${targets[@]}"
}

prepare_c3_runtime() {
  SP_KEEP_DESKTOP_RUNTIME_ARTIFACTS=1 SP_C3_COMPILE_ONLY=1 "${ROOT_DIR}/scripts/launch_ui_c3_desktop.sh" "${jobs}"
}

prepare_python_ui_runtime() {
  SP_KEEP_DESKTOP_RUNTIME_ARTIFACTS=1 SP_C4_COMPILE_ONLY=1 "${ROOT_DIR}/scripts/launch_ui_c4_desktop.sh" "${jobs}"
}

launch_replay() {
  local replay_cmd=(
    "${ROOT_DIR}/tools/replay/replay"
    --headless
    --dcam
    --ecam
  )
  replay_cmd+=("${REPLAY_ARGS[@]}")

  "${replay_cmd[@]}" &
  REPLAY_PID=$!

  sleep 1
  if ! kill -0 "${REPLAY_PID}" >/dev/null 2>&1; then
    wait "${REPLAY_PID}"
    return 1
  fi
}

launch_gpu_param_sync() {
  "${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/tools/replay/onroad_config.py" sync-gpu &
  GPU_SYNC_PID=$!
}

launch_nav_demo() {
  echo "Starting fake nav demo publisher..."
  "${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/tools/replay/fake_nav_demo.py" &
  NAV_PID=$!

  sleep 0.5
  if ! kill -0 "${NAV_PID}" >/dev/null 2>&1; then
    wait "${NAV_PID}"
    return 1
  fi
}

launch_cem_demo() {
  echo "Starting fake CEM demo publisher..."
  "${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/tools/replay/fake_cem_demo.py" &
  CEM_PID=$!

  sleep 0.5
  if ! kill -0 "${CEM_PID}" >/dev/null 2>&1; then
    wait "${CEM_PID}"
    return 1
  fi
}

launch_alert_demo() {
  echo "Starting fake critical alert demo publisher (fires after 20s on-road; blocks replay's selfdriveState)..."
  "${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/tools/replay/fake_alert_demo.py" &
  ALERT_PID=$!

  sleep 0.5
  if ! kill -0 "${ALERT_PID}" >/dev/null 2>&1; then
    wait "${ALERT_PID}"
    return 1
  fi
}

launch_csc_demo() {
  echo "Starting fake CSC demo publisher..."
  "${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/tools/replay/fake_csc_demo.py" &
  CSC_PID=$!

  sleep 0.5
  if ! kill -0 "${CSC_PID}" >/dev/null 2>&1; then
    wait "${CSC_PID}"
    return 1
  fi
}

pick_free_local_port() {
  "${ROOT_DIR}/.venv/bin/python3" - <<'PY'
import socket

# The desktop ZMQ transport hashes replay service names into ports 8023-65535.
# Keep Galaxy below that range so the HTTP server cannot steal a replay service port.
for port in range(4600, 8023):
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
      sock.bind(("127.0.0.1", port))
    except OSError:
      continue
    print(port)
    raise SystemExit(0)

raise SystemExit("Unable to find a free local Galaxy port below the ZMQ replay range.")
PY
}

wait_for_galaxy() {
  local url="${GALAXY_URL}/api/galaxy/status"
  local idx=0

  for ((idx=0; idx<50; idx++)); do
    if ! kill -0 "${GALAXY_PID}" >/dev/null 2>&1; then
      wait "${GALAXY_PID}"
      return 1
    fi

    if "${ROOT_DIR}/.venv/bin/python3" - "${url}" <<'PY' >/dev/null 2>&1; then
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=0.5) as response:
  raise SystemExit(0 if response.status == 200 else 1)
PY
      return 0
    fi

    sleep 0.1
  done

  echo "Timed out waiting for local Galaxy session at ${GALAXY_URL}" >&2
  return 1
}

launch_galaxy() {
  GALAXY_PORT="$(pick_free_local_port)"
  GALAXY_URL="http://127.0.0.1:${GALAXY_PORT}"
  local galaxy_dir="${HOME}/.comma${ONROAD_TEMP_PREFIX}/starpilot/data/galaxy"

  echo "Starting local Galaxy session..."
  (
    export SP_GALAXY_DIR="${galaxy_dir}"
    export SP_GALAXY_HOST="127.0.0.1"
    export SP_GALAXY_PORT="${GALAXY_PORT}"
    export SP_GALAXY_DEBUG=0
    export SP_GALAXY_RELOAD=0
    exec "${ROOT_DIR}/.venv/bin/python3" -m openpilot.starpilot.system.the_galaxy.the_galaxy
  ) &
  GALAXY_PID=$!

  wait_for_galaxy

  echo "Access Galaxy with ${GALAXY_URL}"
}

launch_python_ui() {
  local big="$1"
  (
    export BIG="${big}"
    if [[ "${OFFROAD_DEMO}" == "1" ]]; then
      export PRIME_TYPE=0
    fi
    exec "${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/selfdrive/ui/ui.py"
  ) &
  UI_PIDS+=("$!")
}

configure_bluetooth_demo() {
  case " ${UI_TARGETS[*]-} " in
    *" c3 "*|*" c4 "*)
      local fake_bluetooth="${SP_ONROAD_FAKE_BLUETOOTH:-}"
      if [[ -z "${fake_bluetooth}" ]]; then
        case " ${UI_TARGETS[*]-} " in
          " c3 ") fake_bluetooth="${SP_C3_FAKE_BLUETOOTH:-1}" ;;
          " c4 ") fake_bluetooth="${SP_C4_FAKE_BLUETOOTH:-1}" ;;
          *) fake_bluetooth=1 ;;
        esac
      fi

      if env_var_truthy "${fake_bluetooth}"; then
        export SP_ALLOW_DESKTOP_FAKE_BLUETOOTH=1
      else
        export SP_ALLOW_DESKTOP_FAKE_BLUETOOTH=0
      fi
      ;;
  esac
}

launch_control_bar() {
  local watch_pids="${UI_PIDS[*]}"
  (
    export REPLAY_WATCH_PIDS="${watch_pids}"
    exec "${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/tools/replay/control_bar.py"
  ) &
  UI_PIDS+=("$!")
}

parse_args "$@"

if [[ -n "${LEGACY_UI_SELECTION}" ]]; then
  expand_ui_targets "${LEGACY_UI_SELECTION}"
fi

if [[ "${REPLAY_ONLY}" == "1" ]]; then
  UI_TARGETS=()
else
  dedupe_ui_targets
fi

if [[ "${NAV_DEMO}" == "1" ]]; then
  ensure_nav_demo_replay_blocklist
fi

if [[ "${OFFROAD_DEMO}" == "1" && "${NAV_DEMO}" != "1" ]]; then
  echo "--offroad requires --nav." >&2
  exit 1
fi

if [[ "${LIVE_CONTROLS}" == "1" ]]; then
  ensure_live_controls_replay_blocklist
fi

if [[ "${ALERT_DEMO}" == "1" ]]; then
  ensure_alert_demo_replay_blocklist
fi

if [[ "${CSC_DEMO}" == "1" ]]; then
  ensure_csc_demo_replay_blocklist
fi

if [[ "${GALAXY}" == "1" ]]; then
  ensure_galaxy_replay_blocklist
fi

if [[ ${#REPLAY_ARGS[@]} -eq 0 ]]; then
  usage >&2
  exit 1
fi

prepare_env
trap cleanup EXIT INT TERM

echo "Using OPENPILOT_PREFIX=${OPENPILOT_PREFIX:-<default>}"
if [[ -n "${OPENPILOT_ZMQ_NAMESPACE:-}" ]]; then
  echo "Using OPENPILOT_ZMQ_NAMESPACE=${OPENPILOT_ZMQ_NAMESPACE}"
fi
if [[ -n "${PARAMS_ROOT:-}" ]]; then
  echo "Using PARAMS_ROOT=${PARAMS_ROOT}"
fi

if [[ "${REPLAY_ONLY}" != "1" && "${UI_SELECTION_EXPLICIT}" == "0" && ${#UI_TARGETS[@]} -eq 0 ]]; then
  auto_select_ui_targets
fi

if [[ "${REPLAY_ONLY}" != "1" && ${#UI_TARGETS[@]} -eq 0 ]]; then
  echo "Select at least one UI with --c3, --c4, or use --replay-only." >&2
  exit 1
fi

configure_bluetooth_demo

echo "Preparing replay and desktop UI runtime..."

build_replay

case " ${UI_TARGETS[*]-} " in
  *" c3 "*)
    prepare_c3_runtime
    ;;
esac

case " ${UI_TARGETS[*]-} " in
  *" c4 "*)
    prepare_python_ui_runtime
    ;;
esac

seed_params
seed_starpilot_theme

if [[ "${GALAXY}" == "1" ]]; then
  launch_galaxy
fi

echo "Starting replay: ${REPLAY_ARGS[*]}"
launch_replay
launch_gpu_param_sync

if [[ "${NAV_DEMO}" == "1" && "${OFFROAD_DEMO}" != "1" ]]; then
  launch_nav_demo
fi

if [[ "${CEM_DEMO}" == "1" ]]; then
  launch_cem_demo
fi

if [[ "${ALERT_DEMO}" == "1" ]]; then
  launch_alert_demo
fi

if [[ "${CSC_DEMO}" == "1" ]]; then
  launch_csc_demo
fi

if [[ "${LIVE_CONTROLS}" == "1" ]]; then
  echo "Launching live card and controlsd..."
  "${ROOT_DIR}/.venv/bin/python3" -m selfdrive.car.card &
  LIVE_PIDS+=("$!")
  "${ROOT_DIR}/.venv/bin/python3" -m selfdrive.controls.controlsd &
  LIVE_PIDS+=("$!")
fi

if [[ ${#UI_TARGETS[@]} -eq 0 ]]; then
  echo "Replay is running without UI windows. Press Ctrl-C to stop."
  wait "${REPLAY_PID}"
  exit 0
fi

echo "Launching UIs: ${UI_TARGETS[*]}"

local_target=""
has_raylib=0
for local_target in "${UI_TARGETS[@]}"; do
  case "${local_target}" in
    c3)
      launch_python_ui 1
      has_raylib=1
      ;;
    c4)
      launch_python_ui 0
      has_raylib=1
      ;;
  esac
done

if [[ "${has_raylib}" == "1" ]]; then
  echo "Launching standalone replay controls bar..."
  launch_control_bar
fi

ui_status=0
for pid in "${UI_PIDS[@]-}"; do
  if [[ -n "${pid}" ]]; then
    wait "${pid}" || ui_status=$?
  fi
done
exit "${ui_status}"
