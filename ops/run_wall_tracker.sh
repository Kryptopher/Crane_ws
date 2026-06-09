#!/usr/bin/env bash
# Standalone wall/target tracker (no ROS, no gantry, no dashboard).
#
#   ./ops/run_wall_tracker.sh
#   ./ops/run_wall_tracker.sh -- --detect-width 800 --detect-height 500
#
# Open:
#   http://$(hostname -I | awk '{print $1}'):8090/
#
set -euo pipefail

OAK_IP="${OAK_IP:-192.168.0.153}"
STREAM_PORT="${STREAM_PORT:-8090}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKER="${SCRIPT_DIR}/wall_tracker/wall_tracker.py"
LAYOUT="${SCRIPT_DIR}/wall_tracker/tag_layout.json"

if ! ping -c1 -W2 "${OAK_IP}" &>/dev/null; then
  echo "WARN: OAK ${OAK_IP} not answering ping (tracker may still connect)"
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

echo ""
echo "Starting standalone wall_tracker (MJPEG :${STREAM_PORT})"
echo "  OAK IP: ${OAK_IP}"
echo "  Layout: ${LAYOUT}"
echo "  Stream: http://$(hostname -I | awk '{print $1}'):${STREAM_PORT}/"
echo ""

exec python3 "${TRACKER}" \
  --ip "${OAK_IP}" \
  --layout "${LAYOUT}" \
  --stream-port "${STREAM_PORT}" \
  "$@"
