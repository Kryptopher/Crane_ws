#!/usr/bin/env bash
# Standalone payload tracker test (no gantry, no dashboard).
#
#   ./ops/run_payload_tracker.sh
#   ./ops/run_payload_tracker.sh -- --publish-alpha 0.10
#
# Verify (second terminal):
#   ros2 topic hz /payload/state
#   ros2 topic echo /payload/state --once
#   firefox http://$(hostname -I | awk '{print $1}'):8080/   # annotated MJPEG
#
set -euo pipefail

OAK_IP="${OAK_IP:-192.168.0.153}"
MARKER_SIZE="${MARKER_SIZE:-0.10}"
STREAM_PORT="${STREAM_PORT:-8080}"

cd "$(dirname "$0")/.."
source install/setup.bash

if ! ping -c1 -W2 "${OAK_IP}" &>/dev/null; then
  echo "WARN: OAK ${OAK_IP} not answering ping (tracker may still connect)"
fi

echo "Stopping other payload_tracker instances..."
pkill -f 'payload_perception/payload_tracker' 2>/dev/null || true
sleep 1

echo ""
echo "Starting payload_tracker (ROS + MJPEG :${STREAM_PORT})"
echo "  OAK IP: ${OAK_IP}"
echo "  Topics: /payload/state  /payload/pose  /payload/camera/compressed"
echo ""

exec ros2 run payload_perception payload_tracker -- \
  --ros \
  --ip "${OAK_IP}" \
  --marker-size "${MARKER_SIZE}" \
  --no-wait-motion \
  --ros-publish-hz 50 \
  --stream-port "${STREAM_PORT}" \
  --publish-alpha 0.22 \
  --gantry-publish-alpha 0.24 \
  --publish-max-step-m 0.008 \
  --publish-median 5 \
  --pos-alpha 0.40 \
  --z-min -0.6 \
  --z-max 1.2 \
  "$@"
