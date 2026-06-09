#!/usr/bin/env python3
"""
traj_player — Load velocity CSV into gantry; 100Hz PLAYBACK logs motor encoders during motion.

CSV format (header optional):
  time_s, vx_mm_s, vy_mm_s

Sequence: PROFILE_START → WAYPOINTs (all at once) → PROFILE_DONE
CSV time_s is used during playback after /gantry/enable (t=0 when motors move).

Usage:
  ros2 run gantry_control traj_player.py /path/to/traj.csv
  ros2 run gantry_control traj_player.py --dry-run /path/to/traj.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import SetMode


@dataclass
class VelSegment:
    time_s: float
    vx_mm_s: float
    vy_mm_s: float


@dataclass
class PositionWaypoint:
    time_s: float
    x_m: float
    y_m: float
    vx_mm_s: float
    vy_mm_s: float


def load_velocity_csv(path: Path) -> list[VelSegment]:
    rows: list[VelSegment] = []
    with path.open(newline='') as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row or row[0].strip().startswith('#'):
                continue
            label = row[0].strip().lower()
            if label.startswith('time'):
                continue
            t = float(row[0].strip())
            vx = float(row[1].strip())
            vy = float(row[2].strip())
            rows.append(VelSegment(t, vx, vy))
    if not rows:
        raise ValueError(f'No trajectory rows in {path}')
    return rows


def integrate_waypoints(
    segments: list[VelSegment],
    x0: float,
    y0: float,
    workspace_m: float,
) -> list[PositionWaypoint]:
    waypoints: list[PositionWaypoint] = []
    x, y = x0, y0
    ws = workspace_m

    for i, seg in enumerate(segments):
        if i > 0:
            prev = segments[i - 1]
            dt = seg.time_s - prev.time_s
            if dt <= 0.0:
                raise ValueError(
                    f'Times must be strictly increasing (row {i} → {i + 1})')
            vx_ms = prev.vx_mm_s / 1000.0
            vy_ms = prev.vy_mm_s / 1000.0
            x = max(0.0, min(ws, x + vx_ms * dt))
            y = max(0.0, min(ws, y + vy_ms * dt))
        waypoints.append(PositionWaypoint(
            time_s=seg.time_s, x_m=x, y_m=y,
            vx_mm_s=seg.vx_mm_s, vy_mm_s=seg.vy_mm_s,
        ))

    return waypoints


class TrajPlayer(Node):
    def __init__(self, csv_path: Path, arm_mode: bool, workspace_m: float, dry_run: bool):
        super().__init__('traj_player')
        self._dry_run = dry_run

        segments = load_velocity_csv(csv_path)
        self.get_logger().info(
            f'Loaded {len(segments)} velocity segments from {csv_path}')

        x0, y0, motors_enabled, _ = self._query_gantry_state()
        self._waypoints = integrate_waypoints(segments, x0, y0, workspace_m)
        self.get_logger().info(
            f'Integrated {len(self._waypoints)} knots from start ({x0:.3f}, {y0:.3f}) m')

        for i, wp in enumerate(self._waypoints):
            self.get_logger().info(
                f'  [{i + 1}] t={wp.time_s:.3f}s → ({wp.x_m:.3f}, {wp.y_m:.3f}) m '
                f'v=({wp.vx_mm_s:.1f}, {wp.vy_mm_s:.1f}) mm/s')

        if dry_run:
            return

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self._pub = self.create_publisher(TrajCmd, '/traj_cmd', qos)

        if arm_mode:
            self._arm_traj_mode()

        if motors_enabled:
            self.get_logger().info(
                'Motors enabled — profile will run when PROFILE_DONE is sent')
        else:
            self.get_logger().info(
                'Motors disabled — profile will buffer; call /gantry/enable to move')

    def _query_gantry_state(self) -> tuple[float, float, bool, bool]:
        state = {'x': 0.0, 'y': 0.0, 'enabled': False, 'got': False}

        def cb(msg: GantryState):
            state['x'] = msg.x
            state['y'] = msg.y
            state['enabled'] = msg.enabled
            state['got'] = True

        sub = self.create_subscription(GantryState, '/gantry/state', cb, 10)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if state['got']:
                self.destroy_subscription(sub)
                return state['x'], state['y'], state['enabled'], True
        self.destroy_subscription(sub)
        self.get_logger().warn('No /gantry/state — integrating from (0, 0)')
        return 0.0, 0.0, False, False

    def _arm_traj_mode(self):
        cli = self.create_client(SetMode, '/gantry/set_mode')
        if not cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/gantry/set_mode not available')
        req = SetMode.Request()
        req.mode = 'TRAJ'
        req.csv_path = ''
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.result() or not future.result().success:
            msg = future.result().message if future.result() else 'no response'
            raise RuntimeError(f'Failed to set TRAJ mode: {msg}')
        self.get_logger().info('Gantry in TRAJ mode')

    def _publish(self, wp: PositionWaypoint | None, command: int):
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = command
        if wp is not None:
            msg.time_s = wp.time_s
            msg.x = wp.x_m
            msg.y = wp.y_m
            msg.vx_mm_s = wp.vx_mm_s
            msg.vy_mm_s = wp.vy_mm_s
        self._pub.publish(msg)

    def run(self):
        if self._dry_run:
            self.get_logger().info('Dry run complete (no /traj_cmd published)')
            return

        # Let /traj_cmd subscribers (logger, gantry) match before burst publish.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            n = self._pub.get_subscription_count()
            if n > 0:
                self.get_logger().info(f'/traj_cmd subscribers: {n}')
                break
        else:
            self.get_logger().warn('No /traj_cmd subscribers — publishing anyway')

        self._publish(None, TrajCmd.PROFILE_START)

        for idx, wp in enumerate(self._waypoints):
            self.get_logger().info(
                f'Publish {idx + 1}/{len(self._waypoints)}: '
                f't={wp.time_s:.3f}s v=({wp.vx_mm_s:.1f}, {wp.vy_mm_s:.1f}) mm/s '
                f'pos=({wp.x_m:.3f}, {wp.y_m:.3f}) m')
            self._publish(wp, TrajCmd.WAYPOINT)

        last = self._waypoints[-1]
        self._publish(last, TrajCmd.PROFILE_DONE)
        self.get_logger().info('PROFILE_DONE sent — enable motors to start if not already')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'csv_path',
        help='Velocity trajectory CSV (time_s, vx_mm_s, vy_mm_s)',
    )
    parser.add_argument(
        '--no-arm', action='store_true',
        help='Do not call /gantry/set_mode TRAJ (gantry already in TRAJ)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Integrate and print waypoints only; do not publish',
    )
    parser.add_argument(
        '--workspace-m', type=float, default=1.00,
        help='Workspace size in meters (default 1.00)',
    )
    args = parser.parse_args()

    path = Path(args.csv_path).expanduser()
    if not path.is_file():
        print(f'File not found: {path}', file=sys.stderr)
        return 1

    rclpy.init()
    node = None
    try:
        node = TrajPlayer(
            path,
            arm_mode=not args.no_arm,
            workspace_m=args.workspace_m,
            dry_run=args.dry_run,
        )
        node.run()
    except Exception as exc:
        print(f'[traj_player] {exc}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
