#!/usr/bin/env bash
# Run all test publishers first, then the logger. Ctrl-C stops everything.
set -e

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$WS_DIR/install/setup.bash"

OUT_DIR="${1:-/tmp/logger_test3}"
mkdir -p "$OUT_DIR"

cleanup() {
    echo ""
    echo "Stopping test publishers..."
    kill "$POSE_PID" "$IMU_PID" 2>/dev/null || true
    wait "$POSE_PID" "$IMU_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting /payload/pose_e publisher..."
ros2 topic pub /payload/pose_e std_msgs/msg/Float64MultiArray \
  "{layout: {dim: [{label: 'fields', size: 5, stride: 5}]}, data: [0.0, 1.5, 0.5, 42.0, 10.0]}" \
  --rate 100 > /dev/null 2>&1 &
POSE_PID=$!

echo "Starting /payload/imu_raw publisher..."
ros2 run payload_perception test_publish_imu_raw --rate 100 --duration 120 > /dev/null 2>&1 &
IMU_PID=$!

echo "Waiting 2s for publishers to settle..."
sleep 2

echo "Starting logger (output: $OUT_DIR)..."
ros2 run experiment_logger logger_node \
  --ros-args \
  -p wait_motion_start:=false \
  -p warmup_sec:=0.5 \
  -p output_dir:="$OUT_DIR"
