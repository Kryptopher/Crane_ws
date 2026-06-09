#!/bin/bash
# Systematic crane_ws test runner. Usage: ./test_stack.sh [all|0|1|2|3|4|help]

set -eo pipefail

LOG_DIR="${HOME}/payload_logs"
CRANE_SETUP="${HOME}/crane_ws/install/setup.bash"
OPS_DIR="${HOME}/crane_ws/ops"
PASS=0
FAIL=0

source_ros() {
  set +u
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
  # shellcheck source=/dev/null
  source "${CRANE_SETUP}"
  set -u 2>/dev/null || true
}

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

has_exe() {
  local pkg="$1" exe="$2"
  ros2 pkg executables "${pkg}" 2>/dev/null | awk '{print $2}' | grep -qx "${exe}"
}

# ── Phase 0: environment ─────────────────────────────────────────────
phase0() {
  echo "=== Phase 0: environment ==="
  source_ros

  if ros2 pkg prefix gantry_control &>/dev/null; then
    pass "gantry_control installed"
  else
    fail "gantry_control not found — run: cd ~/crane_ws && colcon build"
  fi

  if ros2 pkg prefix payload_perception &>/dev/null; then
    pass "payload_perception installed"
  else
    fail "payload_perception not found"
  fi

  if ros2 interface show payload_perception_msgs/msg/PayloadState &>/dev/null; then
    pass "PayloadState message available"
  else
    fail "PayloadState message missing"
  fi

  has_exe gantry_control gantry_controller && pass "gantry_controller" || fail "gantry_controller"
  has_exe gantry_control traj_player.py && pass "traj_player.py" || fail "traj_player.py"
  has_exe experiment_logger logger_node && pass "logger_node" || fail "logger_node"
  has_exe payload_perception payload_tracker && pass "payload_tracker" || fail "payload_tracker"
  has_exe payload_perception test_publish_payload_state \
    && pass "test_publish_payload_state" || fail "test_publish_payload_state"
}

# ── Phase 1: traj logging (no USB, no camera) ────────────────────────
phase1() {
  echo ""
  echo "=== Phase 1: traj_cmd logging (no crane USB) ==="
  source_ros
  pkill -f 'experiment_logger/logger_node' 2>/dev/null || true
  sleep 0.5

  ros2 run experiment_logger logger_node &
  local log_pid=$!
  sleep 2

  if ! ros2 run gantry_control traj_player.py --no-arm "${HOME}/csv_profiles/pulse.csv" 2>&1 | grep -q PROFILE_DONE; then
    fail "traj_player did not complete"
    kill "${log_pid}" 2>/dev/null || true
    return
  fi
  sleep 0.3

  local latest
  latest="$(ls -t "${LOG_DIR}"/logger_traj_cmd_*.csv 2>/dev/null | head -1 || true)"
  if [[ -z "${latest}" ]]; then
    fail "no logger_traj_cmd CSV"
    kill "${log_pid}" 2>/dev/null || true
    return
  fi

  local lines
  lines="$(wc -l < "${latest}")"
  if [[ "${lines}" -ge 7 ]]; then
    pass "traj_cmd CSV has ${lines} lines (expected ≥7)"
  else
    fail "traj_cmd CSV only ${lines} lines"
  fi

  kill "${log_pid}" 2>/dev/null || true
  wait "${log_pid}" 2>/dev/null || true
  sleep 1
}

# ── Phase 2: PayloadState + logger (no camera) ───────────────────────
phase2() {
  echo ""
  echo "=== Phase 2: /payload/state + logger (synthetic) ==="
  source_ros
  pkill -f 'experiment_logger/logger_node' 2>/dev/null || true
  sleep 1

  if ! has_exe payload_perception test_publish_payload_state; then
    fail "phase 2 skipped — rebuild payload_perception"
    return
  fi

  ros2 run experiment_logger logger_node --ros-args \
    -p wait_motion_start:=false -p warmup_sec:=0.5 &
  local log_pid=$!
  sleep 2

  ros2 run payload_perception test_publish_payload_state --duration 2 --rate 50 || {
    fail "test_publish_payload_state failed"
    kill "${log_pid}" 2>/dev/null || true
    return
  }
  sleep 1

  local latest
  latest="$(ls -t "${LOG_DIR}"/logger_sync_pose_*.csv 2>/dev/null | head -1 || true)"
  if [[ -z "${latest}" ]]; then
    fail "no logger_sync_pose CSV"
    kill "${log_pid}" 2>/dev/null || true
    return
  fi

  local header
  header="$(head -1 "${latest}")"
  if echo "${header}" | grep -q vx1; then
    pass "sync_pose CSV has velocity columns"
  else
    fail "sync_pose CSV missing vx1 column: ${header}"
  fi

  local data_lines
  data_lines="$(tail -n +2 "${latest}" | wc -l)"
  if [[ "${data_lines}" -ge 20 ]]; then
    pass "sync_pose CSV has ${data_lines} data rows"
  else
    fail "sync_pose CSV only ${data_lines} rows"
  fi

  if tail -n +2 "${latest}" | awk -F, '{if ($5 != "" && $5 != 0) c++} END {exit (c>5)?0:1}'; then
    pass "vx1 column has non-zero samples"
  else
    fail "vx1 column all zero/empty"
  fi

  kill "${log_pid}" 2>/dev/null || true
  wait "${log_pid}" 2>/dev/null || true
}

# ── Phase 3: live topic check (camera required) ────────────────────
phase3() {
  echo ""
  echo "=== Phase 3: live vision (manual — needs OAK @ .153) ==="
  echo "  Run in Terminal 1:"
  echo "    ros2 run payload_perception payload_tracker --ros --ip 192.168.0.153"
  echo "  Run in Terminal 2 (after tags visible):"
  echo "    ros2 topic hz /payload/state"
  echo "    ros2 topic echo /payload/state --once"
  echo "  Pass criteria:"
  echo "    - hz > 10 (typical detect rate)"
  echo "    - valid: true, finite x1/z1/x2/z2"
  echo "    - |vx1| or |vz1| > 0.001 when payload moves"
  echo "  (skipped by automated runner — run manually)"
}

# ── Phase 4: full stack (crane + vision) ─────────────────────────────
phase4() {
  echo ""
  echo "=== Phase 4: full experiment (manual — USB + OAK) ==="
  echo "  Terminal 1:"
  echo "    ros2 launch gantry_control gantry.launch.py oak_ip:=192.168.0.153"
  echo "  Terminal 2:"
  echo "    ros2 run gantry_control traj_player.py ~/csv_profiles/pulse.csv"
  echo "    ros2 run gantry_control gantry_cli.py enable"
  echo "  Pass criteria:"
  echo "    - logger_sync_pose_*.csv has vx1..vz2 columns with data after MOTION_START"
  echo "    - logger_traj_cmd_*.csv has PLAYBACK rows"
  echo "    - |/gantry/state x,y| stay ≤ 1.0 m"
  echo "  (skipped by automated runner — run manually)"
}

# ── main ─────────────────────────────────────────────────────────────
cmd="${1:-all}"
case "$cmd" in
  0) phase0 ;;
  1) phase0; phase1 ;;
  2) phase0; phase2 ;;
  3) phase0; phase3 ;;
  4) phase0; phase4 ;;
  all)
    phase0
    phase1
    phase2
    phase3
    phase4
    ;;
  help|*)
    cat <<EOF
Usage: $0 [all|0|1|2|3|4|help]

  0  Environment + build checks (automated)
  1  Traj log-only: logger + traj_player --no-arm (automated, no USB)
  2  PayloadState: synthetic publisher + logger CSV (automated, no camera)
  3  Live OAK tracker (manual checklist)
  4  Full gantry + vision run (manual checklist)

  all  Run 0, 1, 2 + print 3–4 checklists

Requires: source ~/crane_ws/install/setup.bash (script does this)
EOF
    exit 0
    ;;
esac

echo ""
echo "=== Summary: ${PASS} passed, ${FAIL} failed ==="
[[ "${FAIL}" -eq 0 ]]
