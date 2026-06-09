#!/usr/bin/env bash
# Measure /payload/state, /gantry/state, and optional /payload/pose_e publish rates.
# Run with the stack up (mission_planner or gantry.launch + tracker).
set -euo pipefail

DURATION="${1:-10}"
WINDOW="${2:-50}"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
fi
if [[ -f "${HOME}/crane_ws/install/setup.bash" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/crane_ws/install/setup.bash"
fi

echo "=== Pose topic rates (${DURATION}s window, ros2 --window ${WINDOW}) ==="
echo "Reference: /stack/pose_sync_hz (vision-led EMA of detect FPS)"
echo "Expect /payload/state, /gantry/state, /payload/pose_e within ~3 Hz of sync topic"
echo ""

if ros2 topic list 2>/dev/null | grep -qx '/stack/pose_sync_hz'; then
  echo "--- /stack/pose_sync_hz (reference) ---"
  timeout 3 ros2 topic echo /stack/pose_sync_hz --once 2>&1 || true
  echo ""
fi

_topics=(
  /payload/state
  /gantry/state
)
if ros2 topic list 2>/dev/null | grep -qx '/payload/pose_e'; then
  _topics+=(/payload/pose_e)
fi

for topic in "${_topics[@]}"; do
  echo "--- ${topic} ---"
  timeout "$((DURATION + 3))" ros2 topic hz "$topic" --window "$WINDOW" 2>&1 \
    | head -n 8 || echo "(no messages or topic missing)"
  echo ""
done

echo "=== One-shot samples ==="
ros2 topic echo /gantry/state --once 2>/dev/null | head -n 12 || true
ros2 topic echo /payload/state --once 2>/dev/null | head -n 20 || true
