#!/usr/bin/env python3
"""
logger_node — Record active payload streams to CSV.

- /payload/state (or legacy /payload/pose), /payload/pose_e, and
  /payload/pose_e_rel: continuous streams.
- /traj_cmd: event log → logger_traj_cmd_<ts>.csv
- Synced pose streams → logger_sync_pose_<ts>.csv
- 2+ continuous streams: log rate = min(stream_hz). Single stream: per message.
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import rclpy
from gantry_control.msg import GantryState, TrajCmd
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from payload_perception_msgs.msg import PayloadState
from std_msgs.msg import Float64MultiArray

_TRAJ_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    depth=10,
)

_CMD_NAMES = {
    TrajCmd.WAYPOINT: 'WAYPOINT',
    TrajCmd.PROFILE_DONE: 'PROFILE_DONE',
    TrajCmd.ABORT: 'ABORT',
    TrajCmd.PROFILE_START: 'PROFILE_START',
    TrajCmd.MOTION_START: 'MOTION_START',
    TrajCmd.PLAYBACK: 'PLAYBACK',
    TrajCmd.STREAM: 'STREAM',
}


class LoggerNode(Node):
    def __init__(self):
        super().__init__('logger_node')

        self.declare_parameter('output_dir', os.path.expanduser('~/payload_logs'))
        self.declare_parameter('flush_every', 50)
        self.declare_parameter('warmup_sec', 1.0)
        self.declare_parameter('stale_timeout_sec', 2.0)
        self.declare_parameter('rate_window_sec', 2.0)
        self.declare_parameter('min_log_hz', 1.0)
        self.declare_parameter('max_log_hz', 500.0)
        self.declare_parameter('log_traj_cmd', True)
        self.declare_parameter('wait_motion_start', True)

        self._flush_every = int(self.get_parameter('flush_every').value)
        self._wait_motion = bool(self.get_parameter('wait_motion_start').value)
        self._motion_started = not self._wait_motion
        self._warmup_sec = float(self.get_parameter('warmup_sec').value)
        self._stale_timeout = float(self.get_parameter('stale_timeout_sec').value)
        self._rate_window = float(self.get_parameter('rate_window_sec').value)
        self._min_log_hz = float(self.get_parameter('min_log_hz').value)
        self._max_log_hz = float(self.get_parameter('max_log_hz').value)
        self._log_traj_cmd = bool(self.get_parameter('log_traj_cmd').value)

        out_dir = Path(self.get_parameter('output_dir').value).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self._path = out_dir / f'logger_sync_pose_{ts}.csv'
        self._traj_path = out_dir / f'logger_traj_cmd_{ts}.csv'

        self._started = time.monotonic()
        self._log_start_time: Optional[Time] = None
        self._logging_ready = False

        self._log_pose = False
        self._log_pose_e = False
        self._log_pose_e_rel = False
        self._synced_stream_mode = False

        self._last_pose: Optional[List[float]] = None
        self._last_pose_e: Optional[List[float]] = None
        self._last_pose_e_rel: Optional[List[float]] = None
        self._last_gantry_cart: Optional[List[float]] = None
        self._last_pose_rx = 0.0
        self._last_pose_e_rx = 0.0
        self._last_pose_e_rel_rx = 0.0
        self._last_gantry_rx = 0.0
        self._pose_times: Deque[float] = deque(maxlen=200)
        self._pose_e_times: Deque[float] = deque(maxlen=200)
        self._pose_e_rel_times: Deque[float] = deque(maxlen=200)
        self._last_gantry_mode = ''

        self._fp = None
        self._writer = None
        self._rows = 0
        self._log_timer = None

        self._traj_fp = None
        self._traj_writer = None
        self._traj_rows = 0
        self._motion_start_time: Optional[Time] = None

        self.create_subscription(
            PayloadState, '/payload/state',
            self._on_payload_state, qos_profile_sensor_data)
        self.create_subscription(
            GantryState, '/gantry/state', self._on_gantry_state, 10)
        self.create_subscription(
            Float64MultiArray, '/payload/pose',
            self._on_pose_legacy, qos_profile_sensor_data)
        self.create_subscription(
            Float64MultiArray, '/payload/pose_e',
            self._on_pose_e, qos_profile_sensor_data)
        self.create_subscription(
            Float64MultiArray, '/payload/pose_e_rel',
            self._on_pose_e_rel, qos_profile_sensor_data)
        if self._log_traj_cmd:
            self.create_subscription(
                TrajCmd, '/traj_cmd', self._on_traj_cmd, _TRAJ_QOS)
            self._ensure_traj_log()

        self.create_timer(0.1, self._tick)
        if self._wait_motion:
            self.get_logger().info(
                'Pose CSV waits for MOTION_START (gantry enable) — times synced to motors')
        else:
            self.get_logger().info(
                f'Logger waiting {self._warmup_sec:.1f}s to detect active streams…')
        self.get_logger().info(
            f'Log files this session: {self._path.name}, {self._traj_path.name}')
        if self._log_traj_cmd:
            self.get_logger().info(
                f'Traj CSV ready (empty until /traj_cmd): {self._traj_path}')

    # ── subscriptions ───────────────────────────────────────────────

    def _store_pose_sample(self, motion_t: float, msg_row: List[float]):
        if self._wait_motion and not self._motion_started:
            return
        now = time.monotonic()
        self._last_pose = [motion_t] + msg_row
        self._last_pose_rx = now
        self._pose_times.append(now)
        self._maybe_start_logging()
        if self._logging_ready and self._log_pose and not self._synced_stream_mode:
            self._write_pose_row(now)

    def _on_gantry_state(self, msg: GantryState):
        now = time.monotonic()
        mode = str(msg.mode).upper()
        self._last_gantry_cart = [float(msg.x), float(msg.y)]
        self._last_gantry_rx = now
        if (
            self._wait_motion
            and not self._motion_started
            and mode == 'CSV'
            and self._last_gantry_mode != 'CSV'
            and bool(msg.enabled)
        ):
            self._on_motion_start('CSV_START')
        self._last_gantry_mode = mode

    def _on_payload_state(self, msg: PayloadState):
        row = [
            msg.x1, msg.z1, msg.x2, msg.z2,
            msg.vx1, msg.vz1, msg.vx2, msg.vz2,
            msg.cam_x, msg.cam_y, msg.cam_z,
            msg.gantry_x, msg.gantry_y, msg.gantry_z,
        ]
        self._store_pose_sample(msg.motion_time_sec, row)

    def _on_pose_legacy(self, msg: Float64MultiArray):
        if len(msg.data) >= 9:
            self._store_pose_sample(
                msg.data[0],
                list(msg.data[1:9]),
            )
        elif len(msg.data) >= 5:
            self._store_pose_sample(
                msg.data[0],
                list(msg.data[1:5]) + [float('nan')] * 4,
            )

    def _on_pose_e(self, msg: Float64MultiArray):
        if len(msg.data) < 5:
            return
        if self._wait_motion and not self._motion_started:
            return
        now = time.monotonic()
        self._last_pose_e = list(msg.data[:5])
        self._last_pose_e_rx = now
        self._pose_e_times.append(now)
        self._maybe_start_logging()
        if self._logging_ready and self._log_pose_e and not self._synced_stream_mode:
            self._write_pose_row(now)

    def _on_pose_e_rel(self, msg: Float64MultiArray):
        if len(msg.data) < 9:
            return
        if self._wait_motion and not self._motion_started:
            return
        now = time.monotonic()
        self._last_pose_e_rel = list(msg.data[:9])
        self._last_pose_e_rel_rx = now
        self._pose_e_rel_times.append(now)
        self._maybe_start_logging()
        if self._logging_ready and self._log_pose_e_rel and not self._synced_stream_mode:
            self._write_pose_row(now)

    def _on_traj_cmd(self, msg: TrajCmd):
        self._ensure_traj_log()
        t_sec = self._stamp_to_session_sec(msg.header.stamp)
        cmd = _CMD_NAMES.get(msg.command, str(msg.command))

        if msg.command == TrajCmd.MOTION_START:
            self._on_motion_start()
            motion_t = '0.000000'
        elif self._motion_start_time is not None:
            motion_t = (
                f'{(self.get_clock().now() - self._motion_start_time).nanoseconds * 1e-9:.6f}'
            )
        else:
            motion_t = ''

        self._traj_writer.writerow([
            f'{t_sec:.6f}',
            motion_t,
            f'{msg.time_s:.6f}',
            f'{msg.x:.6f}',
            f'{msg.y:.6f}',
            f'{msg.vx_mm_s:.3f}',
            f'{msg.vy_mm_s:.3f}',
            f'{msg.motor_a_pos_rad:.6f}',
            f'{msg.motor_b_pos_rad:.6f}',
            f'{msg.motor_a_vel_rad_s:.6f}',
            f'{msg.motor_b_vel_rad_s:.6f}',
            cmd,
        ])
        self._traj_rows += 1
        self._traj_fp.flush()
        self.get_logger().info(
            f'TRAJ cmd logged: t={t_sec:.3f}s ({msg.x:.3f}, {msg.y:.3f}) '
            f'v=({msg.vx_mm_s:.1f},{msg.vy_mm_s:.1f}) {cmd}',
            throttle_duration_sec=1.0)

    # ── stream detection / rates ──────────────────────────────────────

    def _stamp_to_session_sec(self, stamp) -> float:
        t_msg = Time.from_msg(stamp)
        if self._log_start_time is None:
            self._log_start_time = self.get_clock().now()
        return (t_msg - self._log_start_time).nanoseconds * 1e-9

    def _continuous_stream_count(self, now: float) -> int:
        n = 0
        if self._log_pose and self._pose_active(now):
            n += 1
        if self._log_pose_e and self._pose_e_active(now):
            n += 1
        if self._log_pose_e_rel and self._pose_e_rel_active(now):
            n += 1
        return n

    def _pose_active(self, now: float) -> bool:
        return (
            self._last_pose is not None
            and (now - self._last_pose_rx) <= self._stale_timeout
        )

    def _pose_e_active(self, now: float) -> bool:
        return (
            self._last_pose_e is not None
            and (now - self._last_pose_e_rx) <= self._stale_timeout
        )

    def _pose_e_rel_active(self, now: float) -> bool:
        return (
            self._last_pose_e_rel is not None
            and (now - self._last_pose_e_rel_rx) <= self._stale_timeout
        )

    def _estimate_hz(self, times: Deque[float], now: float) -> float:
        while times and (now - times[0]) > self._rate_window:
            times.popleft()
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        if span <= 0.0:
            return 0.0
        return (len(times) - 1) / span

    def _clamp_hz(self, hz: float) -> float:
        if hz <= 0.0:
            return self._min_log_hz
        return max(self._min_log_hz, min(hz, self._max_log_hz))

    def _on_motion_start(self, trigger: str = 'MOTION_START'):
        if self._motion_started:
            return
        self._motion_started = True
        self._motion_start_time = self.get_clock().now()
        self._log_start_time = self._motion_start_time
        self._last_pose = None
        self._last_pose_e = None
        self._last_pose_e_rel = None
        self._pose_times.clear()
        self._pose_e_times.clear()
        self._pose_e_rel_times.clear()
        self._started = time.monotonic()
        self.get_logger().info(
            f'{trigger} — experiment pose log t=0 synced to motors')
        self._begin_logging()

    def _maybe_start_logging(self):
        if self._logging_ready:
            return
        if self._wait_motion and not self._motion_started:
            return
        if time.monotonic() - self._started < self._warmup_sec:
            return
        self._begin_logging()

    def _tick(self):
        now = time.monotonic()
        if not self._logging_ready:
            self._maybe_start_logging()
            return
        if not self._synced_stream_mode:
            return
        self._update_log_timer(now)

    def _ensure_traj_log(self):
        if self._traj_writer is not None:
            return
        if self._log_start_time is None:
            self._log_start_time = self.get_clock().now()
        self._traj_fp = open(self._traj_path, 'w', newline='')
        self._traj_writer = csv.writer(self._traj_fp)
        self._traj_writer.writerow([
            'log_time_sec', 'motion_time_sec', 'profile_time_s', 'traj_x_m',
            'traj_y_m', 'vx_mm_s', 'vy_mm_s',
            'motor_a_pos_rad', 'motor_b_pos_rad',
            'motor_a_vel_rad_s', 'motor_b_vel_rad_s', 'command',
        ])
        self._traj_fp.flush()
        self.get_logger().info(f'TRAJ command log → {self._traj_path}')

    def _begin_logging(self):
        now = time.monotonic()
        self._log_pose = self._pose_active(now)
        self._log_pose_e = self._pose_e_active(now)
        self._log_pose_e_rel = self._pose_e_rel_active(now)

        if not self._log_pose and not self._log_pose_e and not self._log_pose_e_rel:
            self.get_logger().warn(
                'No active pose streams after warmup — still waiting…')
            self._started = now
            return

        if self._log_start_time is None:
            self._log_start_time = self.get_clock().now()

        self._fp = open(self._path, 'w', newline='')
        self._writer = csv.writer(self._fp)
        header = ['time_sec']
        if self._log_pose:
            header.extend([
                'x1', 'z1', 'x2', 'z2',
                'vx1', 'vz1', 'vx2', 'vz2',
                'cam_x', 'cam_y', 'cam_z',
                'gantry_x', 'gantry_y', 'gantry_z',
                'cart_x', 'cart_y',
            ])
        if self._log_pose_e:
            header.extend(['pitch_deg', 'roll_deg', 'pitch_count', 'roll_count'])
        if self._log_pose_e_rel:
            header.extend([
                'pitch_rel_deg', 'roll_rel_deg',
                'x_rel_m', 'y_rel_m', 'z_rel_m',
                'vx_rel_m_s', 'vy_rel_m_s', 'vz_rel_m_s',
            ])
        self._writer.writerow(header)
        self._logging_ready = True

        n_streams = int(self._log_pose) + int(self._log_pose_e) + int(self._log_pose_e_rel)
        self._synced_stream_mode = n_streams >= 2

        streams = []
        if self._log_pose:
            streams.append('vision (/payload/state)')
        if self._log_pose_e:
            streams.append('encoder (/payload/pose_e)')
        if self._log_pose_e_rel:
            streams.append('encoder relative (/payload/pose_e_rel)')

        if self._synced_stream_mode:
            hz = self._matched_log_hz(now)
            self._reset_log_timer(1.0 / hz)
            self.get_logger().info(
                f'Logging {", ".join(streams)} @ {hz:.1f} Hz (min rate) → {self._path}')
        else:
            hz = self._matched_log_hz(now)
            self.get_logger().info(
                f'Logging {streams[0]} @ ~{hz:.1f} Hz (per message) → {self._path}')

        if self._log_traj_cmd:
            self.get_logger().info(
                f'TRAJ commands → {self._traj_path} (on each /traj_cmd)')

    def _matched_log_hz(self, now: float) -> float:
        rates = []
        if self._log_pose:
            rates.append(self._clamp_hz(self._estimate_hz(self._pose_times, now)))
        if self._log_pose_e:
            rates.append(self._clamp_hz(self._estimate_hz(self._pose_e_times, now)))
        if self._log_pose_e_rel:
            rates.append(self._clamp_hz(self._estimate_hz(self._pose_e_rel_times, now)))
        if not rates:
            return self._min_log_hz
        return min(rates)

    def _reset_log_timer(self, period_sec: float):
        if self._log_timer is not None:
            self._log_timer.cancel()
        self._log_timer = self.create_timer(float(period_sec), self._on_log_timer)

    def _update_log_timer(self, now: float):
        if not self._synced_stream_mode:
            return
        hz = self._matched_log_hz(now)
        period = 1.0 / hz
        if self._log_timer is None:
            self._reset_log_timer(period)
            return
        current = self._log_timer.timer_period_ns * 1e-9
        if abs(current - period) / period > 0.05:
            self._reset_log_timer(period)

    def _on_log_timer(self):
        now = time.monotonic()
        if not self._logging_ready or not self._synced_stream_mode:
            return
        if self._continuous_stream_count(now) < 2:
            return
        self._write_pose_row(now)

    # ── CSV output ────────────────────────────────────────────────────

    def _write_pose_row(self, now: float):
        if self._writer is None:
            return

        include_pose = self._log_pose and self._pose_active(now)
        include_pose_e = self._log_pose_e and self._pose_e_active(now)
        include_pose_e_rel = self._log_pose_e_rel and self._pose_e_rel_active(now)
        if not include_pose and not include_pose_e and not include_pose_e_rel:
            return

        v = self._last_pose if include_pose else None
        e = self._last_pose_e if include_pose_e else None
        er = self._last_pose_e_rel if include_pose_e_rel else None

        def _fmt_pose(v: List[float]) -> List[str]:
            out = [f'{v[0]:.6f}']
            for i in range(1, len(v)):
                val = v[i]
                out.append(
                    f'{val:.6f}' if isinstance(val, (int, float)) and math.isfinite(val)
                    else ''
                )
            return out

        def _cart_cols() -> List[str]:
            if self._last_gantry_cart is None:
                return ['', '']
            return [f'{self._last_gantry_cart[0]:.6f}', f'{self._last_gantry_cart[1]:.6f}']

        if self._log_pose:
            if v is None:
                return
            row = _fmt_pose(v)
            row.extend(_cart_cols())
        else:
            source_t = e[0] if e is not None else er[0]
            row = [f'{source_t:.6f}']

        if self._log_pose_e:
            if e is None:
                return
            row.extend([f'{e[1]:.6f}', f'{e[2]:.6f}', f'{e[3]:.1f}', f'{e[4]:.1f}'])

        if self._log_pose_e_rel:
            if er is None:
                return
            row.extend([
                f'{er[1]:.6f}', f'{er[2]:.6f}',
                f'{er[3]:.6f}', f'{er[4]:.6f}', f'{er[5]:.6f}',
                f'{er[6]:.6f}', f'{er[7]:.6f}', f'{er[8]:.6f}',
            ])

        if not self._log_pose and not self._log_pose_e and self._log_pose_e_rel:
            # row already contains pose_e_rel time and columns.
            pass
        elif not self._log_pose and self._log_pose_e and not self._log_pose_e_rel:
            if e is None:
                return
            row = [f'{e[0]:.6f}']
            row.extend([f'{e[1]:.6f}', f'{e[2]:.6f}', f'{e[3]:.1f}', f'{e[4]:.1f}'])

        self._writer.writerow(row)
        self._rows += 1
        if self._rows % self._flush_every == 0:
            self._fp.flush()

    def destroy_node(self):
        if self._log_timer is not None:
            self._log_timer.cancel()
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self.get_logger().info(f'Wrote {self._rows} rows to {self._path}')
        else:
            self.get_logger().info('No pose rows written (no active streams)')
        if self._traj_fp is not None:
            self._traj_fp.flush()
            self._traj_fp.close()
            self.get_logger().info(
                f'Wrote {self._traj_rows} traj commands to {self._traj_path}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LoggerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
