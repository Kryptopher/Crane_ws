#!/bin/bash
# Traj + logger workflow (Jetson / crane_ws). Run: ./run_traj_log.sh help

set -eo pipefail

JETSON_IP="${JETSON_IP:-192.168.0.101}"
CSV_DEFAULT="${HOME}/csv_profiles/pulse.csv"
LOG_DIR="${HOME}/payload_logs"
CRANE_SETUP="${HOME}/crane_ws/install/setup.bash"

source_ros() {
  set +u
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
  if [[ ! -f "${CRANE_SETUP}" ]]; then
    echo "ERROR: ${CRANE_SETUP} missing."
    echo "Build: cd ~/crane_ws && colcon build"
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CRANE_SETUP}"
  set -u 2>/dev/null || true
}

source_ros

cmd="${1:-help}"
shift || true

case "$cmd" in
  setup)
    echo "Paste this in EVERY new SSH terminal before other commands:"
    echo ""
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source ~/crane_ws/install/setup.bash"
    echo ""
    ;;
  launch)
    echo "Starting gantry + logger → logs in ${LOG_DIR}"
    echo "Requires motor USB hub: ls /dev/ttyXRUSB*"
    if ! ls /dev/ttyXRUSB* >/dev/null 2>&1; then
      echo "WARNING: No /dev/ttyXRUSB* — gantry will fail. For log-only test use: $0 logonly"
    fi
    exec ros2 launch gantry_control traj_log.launch.py
    ;;
  logonly)
    echo "Logger only (no crane USB needed) → ${LOG_DIR}"
    echo "In another terminal: $0 play --no-arm [csv]"
    exec ros2 run experiment_logger logger_node
    ;;
  play)
    NO_ARM=""
    if [[ "${1:-}" == "--no-arm" ]]; then
      NO_ARM="--no-arm"
      shift
    fi
    CSV="${1:-$CSV_DEFAULT}"
    echo "Loading trajectory: ${CSV}"
    if [[ -n "${NO_ARM}" ]]; then
      ros2 run gantry_control traj_player.py --no-arm "${CSV}"
    else
      ros2 run gantry_control traj_player.py "${CSV}"
    fi
    echo ""
    if [[ -n "${NO_ARM}" ]]; then
      echo "Done (log-only). Check: $0 show"
    else
      echo "Done. If motors were off, run: $0 enable"
    fi
    ;;
  show)
    mkdir -p "${LOG_DIR}"
    echo "=== ${LOG_DIR} (newest traj_cmd logs) ==="
    ls -lt "${LOG_DIR}"/logger_traj_cmd_*.csv "${LOG_DIR}"/*_traj_cmd.csv 2>/dev/null | head -5 || echo "(no traj_cmd CSV yet)"
    LATEST="$(ls -t "${LOG_DIR}"/logger_traj_cmd_*.csv 2>/dev/null | head -1 || true)"
    if [[ -z "${LATEST}" ]]; then
      LATEST="$(ls -t "${LOG_DIR}"/*_traj_cmd.csv 2>/dev/null | head -1 || true)"
    fi
    if [[ -n "${LATEST}" ]]; then
      echo ""
      echo "Latest file: ${LATEST}"
      echo "Lines: $(wc -l < "${LATEST}")"
      head -3 "${LATEST}"
    fi
    ;;
  enable)
    ros2 run gantry_control gantry_cli.py enable
    ;;
  scp)
    DEST="${2:-./jetson_logs}"
    HOST="${1:-sanjay@${JETSON_IP}}"
    mkdir -p "${DEST}"
    echo "Fetching traj_cmd CSV from ${HOST}:${LOG_DIR}/"
    scp "${HOST}:${LOG_DIR}/*_traj_cmd.csv" "${DEST}/" || {
      echo "scp failed. On Jetson run: $0 show"
      exit 1
    }
    echo "Saved to ${DEST}/"
    ls -la "${DEST}/"*_traj_cmd.csv 2>/dev/null | tail -5
    ;;
  help|*)
    cat <<EOF
=== REQUIRED: every new SSH terminal ===
  source /opt/ros/humble/setup.bash
  source ~/crane_ws/install/setup.bash
  cd ~/crane_ws/ops

=== A) Log traj_cmd only (no crane USB) ===
  Terminal 1: $0 logonly
  Terminal 2: $0 play --no-arm ~/csv_profiles/pulse.csv
  Terminal 2: $0 show

=== B) Full run (crane USB plugged in) ===
  Terminal 1: $0 launch
  Terminal 2: $0 play ~/csv_profiles/pulse.csv
  Terminal 2: $0 enable
  Terminal 2: $0 show

=== Download from Windows PC ===
  scp sanjay@${JETSON_IP}:~/payload_logs/*_traj_cmd.csv ./jetson_logs/

Default CSV: ${CSV_DEFAULT}
EOF
    ;;
esac
