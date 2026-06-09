#!/usr/bin/env python3
"""
adaptive_id_player.py

ROS 2 adaptive system-identification test for gantry crane payload.

Identification only (no adaptive control yet).

Workflow:
  1. Arms gantry TRAJ mode (unless --no-arm).
  2. Publishes constant-velocity profile on /traj_cmd:
        PROFILE_START -> WAYPOINT(t=0) -> WAYPOINT(t=duration) -> PROFILE_DONE
  3. Operator enables motors (/gantry/enable or UI).
  4. gantry_controller publishes MOTION_START on /traj_cmd; this node resets I1/I2/I3.
  5. Subscribes to payload x(t) from tracker; runs Eq. (21) online.
  6. Prints cond(B), omega_n, zeta, T, A0, A1, A2.

Payload topics (tripod-mounted OAK — no cart compensation):
  /payload/state  PayloadState — use gantry_x for swing (default)
  /payload/pose   legacy Float64MultiArray (camera x1/z1 per tag)

Use profile motion time (not ROS wall clock) for Eq. (21) when available.

Published estimates:
  /adaptive_id/estimate  Float64MultiArray:
      [id_t, x_rel_mm, condB, omega_n, freq_hz, zeta, T, A0, A1, A2]
  /adaptive_id/zv_T      Float64 candidate shaper period T
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import PointStamped, PoseStamped

from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import SetMode

try:
    from payload_perception_msgs.msg import PayloadState
except ImportError:
    PayloadState = None  # type: ignore


@dataclass
class Estimate:
    t: float
    x: float
    i1: float
    i2: float
    i3: float
    cond_b: float
    omega_n: float
    zeta: float
    T: float
    A0: float
    A1: float
    A2: float


class AdaptiveIdentifier:
    """
    Online implementation of Eq. (21a)-(21b).

    I1 = integral of x
    I2 = double integral of x
    I3 = triple integral of x

    Units: x and K in mm, t in seconds (profile / motion time).
    """

    def __init__(
        self,
        K_mm_s: float,
        A0: float,
        cond_threshold: float,
        omega_min: float,
        omega_max: float,
        zeta_min: float,
        zeta_max: float,
    ):
        self.K = float(K_mm_s)
        self.A0 = float(A0)
        self.cond_threshold = float(cond_threshold)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.zeta_min = float(zeta_min)
        self.zeta_max = float(zeta_max)
        self.reset()

    def reset(self):
        self.t0: float | None = None
        self.prev_t: float | None = None
        self.prev_x: float | None = None
        self.x_zero = 0.0
        self.I1 = 0.0
        self.I2 = 0.0
        self.I3 = 0.0
        self.latest_valid: Estimate | None = None

    def start(self, t_motion: float, x_now_mm: float, zero_at_start: bool = True):
        """Reset integrators at motion start (t=0 at MOTION_START)."""
        self.reset()
        self.x_zero = float(x_now_mm) if zero_at_start else 0.0
        x_rel_mm = float(x_now_mm) - self.x_zero
        self.t0 = float(t_motion)
        self.prev_t = float(t_motion)
        self.prev_x = x_rel_mm

    def update(self, t_motion: float, x_now_mm: float) -> Estimate | None:
        if self.t0 is None:
            return None

        if self.prev_t is None or self.prev_x is None:
            self.prev_t = t_motion
            self.prev_x = x_now_mm
            return None

        t = t_motion - self.t0
        dt = t_motion - self.prev_t
        x_rel_mm = float(x_now_mm) - self.x_zero

        if dt <= 0.0 or t <= 0.0:
            return None

        I1_old = self.I1
        I2_old = self.I2
        x_avg = 0.5 * (self.prev_x + x_rel_mm)

        self.I1 += x_avg * dt
        self.I2 += 0.5 * (I1_old + self.I1) * dt
        self.I3 += 0.5 * (I2_old + self.I2) * dt

        self.prev_t = t_motion
        self.prev_x = x_rel_mm

        I1 = self.I1
        I2 = self.I2
        I3 = self.I3
        K = self.K

        B = np.array(
            [
                [I1, I2 - K * t**3 / 6.0],
                [I2, I3 - K * t**4 / 24.0],
            ],
            dtype=float,
        )
        rhs = np.array([-x_rel_mm, -I1], dtype=float)

        try:
            cond_b = float(np.linalg.cond(B))
        except np.linalg.LinAlgError:
            return None

        if not math.isfinite(cond_b) or cond_b > self.cond_threshold:
            return None

        try:
            theta = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return None

        two_zeta_omega = float(theta[0])
        omega_sq = float(theta[1])

        if not math.isfinite(omega_sq) or omega_sq <= 0.0:
            return None

        omega_n = math.sqrt(omega_sq)
        zeta = two_zeta_omega / (2.0 * omega_n)

        if not math.isfinite(omega_n) or not math.isfinite(zeta):
            return None

        if not (self.omega_min <= omega_n <= self.omega_max):
            return None

        if not (self.zeta_min <= zeta <= self.zeta_max):
            return None

        shaper = self.compute_shaper(zeta, omega_n)
        if shaper is None:
            return None

        T, A0, A1, A2 = shaper
        est = Estimate(
            t=t,
            x=x_rel_mm,
            i1=I1,
            i2=I2,
            i3=I3,
            cond_b=cond_b,
            omega_n=omega_n,
            zeta=zeta,
            T=T,
            A0=A0,
            A1=A1,
            A2=A2,
        )
        self.latest_valid = est
        return est

    def compute_shaper(self, zeta: float, omega_n: float):
        if omega_n <= 0.0:
            return None
        if zeta < 0.0 or zeta >= 1.0:
            return None

        wd_factor = math.sqrt(max(1.0 - zeta**2, 1.0e-12))
        T = math.pi / (omega_n * wd_factor)

        exp1 = math.exp(math.pi * zeta / wd_factor)
        exp2 = math.exp(2.0 * math.pi * zeta / wd_factor)

        A0 = self.A0
        A1 = (A0 + (1.0 - A0) * exp2) / (exp1 + exp2)
        A2 = 1.0 - A0 - A1
        return T, A0, A1, A2


class AdaptiveIDPlayer(Node):
    def __init__(self, args):
        super().__init__('adaptive_id_player')

        self.args = args
        self.profile_sent = False
        self.motion_started = False
        self.pending_motion_start = False

        self.latest_gantry_state: GantryState | None = None
        self.latest_payload_x_mm: float | None = None
        self.latest_motion_t: float | None = None
        self.csv_file = None
        self.csv_writer: csv.writer | None = None

        self.identifier = AdaptiveIdentifier(
            K_mm_s=args.vx_mm_s,
            A0=args.a0,
            cond_threshold=args.cond_threshold,
            omega_min=args.omega_min,
            omega_max=args.omega_max,
            zeta_min=args.zeta_min,
            zeta_max=args.zeta_max,
        )

        traj_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.traj_pub = self.create_publisher(TrajCmd, '/traj_cmd', traj_qos)
        self.estimate_pub = self.create_publisher(
            Float64MultiArray, args.estimate_topic, 10)
        self.zv_t_pub = self.create_publisher(Float64, args.zv_t_topic, 10)

        if args.log_csv:
            self._open_csv_log(args.log_csv)

        self.create_subscription(
            GantryState, '/gantry/state', self._gantry_state_cb, 10)

        self.create_subscription(
            TrajCmd, '/traj_cmd', self._traj_cmd_cb, traj_qos)

        self._create_payload_subscription()

        if not args.no_arm:
            self._arm_traj_mode()

        self.startup_timer = self.create_timer(0.5, self._startup_timer_cb)

        self.get_logger().info('adaptive_id_player started')
        self.get_logger().info(f'Payload topic: {args.payload_topic}')
        self.get_logger().info(f'Payload type:  {args.payload_type}')
        self.get_logger().info(f'Payload units: {args.payload_units}')
        if args.payload_type == 'payloadstate':
            self.get_logger().info(f'Swing field:   {args.swing_field}')
        else:
            self.get_logger().info(f'Pose x index:  {args.pose_x_index}')
        self.get_logger().info(
            f'ID profile: vx={args.vx_mm_s:.1f} mm/s, '
            f'vy={args.vy_mm_s:.1f} mm/s, duration={args.duration:.1f} s')
        self.get_logger().info(f'Estimate topic: {args.estimate_topic}')
        self.get_logger().info(f'ZV T topic:      {args.zv_t_topic}')
        self.get_logger().info(
            'System ID starts on MOTION_START (/traj_cmd) after /gantry/enable')

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        super().destroy_node()

    def _open_csv_log(self, path_arg: str):
        path = Path(path_arg).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = path.open('w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'row_type',
            'wall_time_sec',
            'motion_time_sec',
            'id_time_sec',
            'raw_x_mm',
            'x_rel_mm',
            'I1_mm_s',
            'I2_mm_s2',
            'I3_mm_s3',
            'cond_b',
            'omega_n_rad_s',
            'freq_hz',
            'zeta',
            'T_sec',
            'A0',
            'A1',
            'A2',
        ])
        self.csv_file.flush()
        self.get_logger().info(f'CSV log: {path}')

    def _create_payload_subscription(self):
        topic = self.args.payload_topic
        msg_type = self.args.payload_type
        qos = qos_profile_sensor_data

        if msg_type == 'float64':
            self.create_subscription(
                Float64, topic, self._payload_float64_cb, qos)
        elif msg_type == 'float64multiarray':
            self.create_subscription(
                Float64MultiArray, topic, self._payload_float64multiarray_cb, qos)
        elif msg_type == 'payloadstate':
            if PayloadState is None:
                raise RuntimeError(
                    'payload_perception_msgs not built; use float64multiarray')
            self.create_subscription(
                PayloadState, topic, self._payload_state_cb, qos)
        elif msg_type == 'pointstamped':
            self.create_subscription(
                PointStamped, topic, self._payload_pointstamped_cb, qos)
        elif msg_type == 'posestamped':
            self.create_subscription(
                PoseStamped, topic, self._payload_posestamped_cb, qos)
        else:
            raise ValueError(
                'Unsupported --payload-type. Use: float64, float64multiarray, '
                'payloadstate, pointstamped, posestamped')

    def _gantry_state_cb(self, msg: GantryState):
        self.latest_gantry_state = msg

    def _traj_cmd_cb(self, msg: TrajCmd):
        if msg.command != TrajCmd.MOTION_START:
            return
        if self.motion_started and self.identifier.t0 is not None:
            return

        self.motion_started = True
        self.pending_motion_start = True

        if self.latest_payload_x_mm is not None and self.latest_motion_t is not None:
            self._begin_id(self.latest_motion_t, self.latest_payload_x_mm)
        else:
            self.get_logger().warn(
                'MOTION_START: waiting for first payload sample to start ID')

    def _begin_id(self, t_motion: float, x_mm: float):
        self.pending_motion_start = False
        self.identifier.start(
            t_motion, x_mm, zero_at_start=self.args.zero_at_start)
        zero_text = f' zero={self.identifier.x_zero:.2f} mm'
        self.get_logger().info(
            f'MOTION_START: ID started at t={t_motion:.3f}s '
            f'x={x_mm:.2f} mm{zero_text}')

    def _startup_timer_cb(self):
        if self.profile_sent:
            return
        self.profile_sent = True
        try:
            self.startup_timer.cancel()
        except Exception:
            pass

        if self.args.dry_run:
            self.get_logger().warn('Dry run: not publishing /traj_cmd')
            if self.args.start_immediately:
                self._start_id_immediately()
            return

        self._publish_constant_velocity_profile()

        if self.args.start_immediately:
            self._start_id_immediately()

    def _start_id_immediately(self):
        if self.latest_payload_x_mm is None:
            self.get_logger().warn('--start-immediately: no payload sample yet')
            return
        t = self.latest_motion_t if self.latest_motion_t is not None else 0.0
        self.motion_started = True
        self._begin_id(t, self.latest_payload_x_mm)
        self.get_logger().warn('ID started immediately (debug; not MOTION_START sync)')

    def _arm_traj_mode(self):
        cli = self.create_client(SetMode, '/gantry/set_mode')
        if not cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/gantry/set_mode not available')

        req = SetMode.Request()
        req.mode = 'TRAJ'
        req.csv_path = ''

        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not result.success:
            msg = result.message if result is not None else 'no response'
            raise RuntimeError(f'Failed to set TRAJ mode: {msg}')

        self.get_logger().info('Gantry set to TRAJ mode')

    def _publish(
        self,
        command: int,
        time_s: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
    ):
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = int(command)
        msg.time_s = float(time_s)

        if self.latest_gantry_state is not None:
            x0 = float(self.latest_gantry_state.x)
            y0 = float(self.latest_gantry_state.y)
        else:
            x0 = 0.0
            y0 = 0.0

        msg.x = x0 + (vx / 1000.0) * time_s
        msg.y = y0 + (vy / 1000.0) * time_s
        msg.vx_mm_s = float(vx)
        msg.vy_mm_s = float(vy)
        self.traj_pub.publish(msg)

    def _publish_constant_velocity_profile(self):
        vx = self.args.vx_mm_s
        vy = self.args.vy_mm_s
        duration = self.args.duration

        self.get_logger().info('Publishing constant-velocity ID profile')
        self._publish(TrajCmd.PROFILE_START)
        self._publish(TrajCmd.WAYPOINT, time_s=0.0, vx=vx, vy=vy)
        self._publish(TrajCmd.WAYPOINT, time_s=duration, vx=vx, vy=vy)
        self._publish(TrajCmd.PROFILE_DONE, time_s=duration, vx=0.0, vy=0.0)

        self.get_logger().info(
            'PROFILE_DONE — enable motors to run (UI Enable or gantry_cli enable)')

    def _payload_float64_cb(self, msg: Float64):
        self._handle_payload_x(float(msg.data), None)

    def _payload_float64multiarray_cb(self, msg: Float64MultiArray):
        if len(msg.data) <= self.args.pose_x_index:
            self.get_logger().warn(
                f'Payload array too short for index {self.args.pose_x_index}')
            return
        t_motion = float(msg.data[0]) if len(msg.data) > 0 else None
        x = float(msg.data[self.args.pose_x_index])
        self._handle_payload_x(x, t_motion)

    def _payload_state_cb(self, msg: 'PayloadState'):
        if not msg.valid:
            return
        field = self.args.swing_field
        try:
            x = float(getattr(msg, field))
        except AttributeError:
            self.get_logger().error(f'Invalid --swing-field {field!r}')
            return
        if not math.isfinite(x):
            return
        self._handle_payload_x(x, float(msg.motion_time_sec))

    def _payload_pointstamped_cb(self, msg: PointStamped):
        self._handle_payload_x(float(msg.point.x), None)

    def _payload_posestamped_cb(self, msg: PoseStamped):
        self._handle_payload_x(float(msg.pose.position.x), None)

    def _handle_payload_x(self, raw_x: float, t_motion: float | None):
        if self.args.payload_units == 'm':
            x_mm = raw_x * 1000.0
        elif self.args.payload_units == 'mm':
            x_mm = raw_x
        else:
            raise ValueError('--payload-units must be m or mm')

        x_mm *= self.args.payload_sign
        x_mm -= self.args.payload_offset_mm

        self.latest_payload_x_mm = x_mm
        if t_motion is not None:
            self.latest_motion_t = float(t_motion)

        if self.pending_motion_start and t_motion is not None:
            self._begin_id(t_motion, x_mm)
            return

        if not self.motion_started:
            self._maybe_print_not_started()
            return

        if t_motion is None:
            self.get_logger().warn_throttle(
                5.0, 'No motion_time in payload; use /payload/pose or /payload/state')
            return

        if self.identifier.t0 is None:
            self._begin_id(t_motion, x_mm)
            return

        self._log_sample(t_motion, x_mm)
        est = self.identifier.update(t_motion, x_mm)
        if est is None:
            self._maybe_print_waiting()
            return

        self._publish_estimate(est)
        self._log_estimate(t_motion, x_mm, est)
        self._maybe_print_estimate(est)

    def _publish_estimate(self, est: Estimate):
        freq_hz = est.omega_n / (2.0 * math.pi)

        msg = Float64MultiArray()
        msg.data = [
            est.t,
            est.x,
            est.cond_b,
            est.omega_n,
            freq_hz,
            est.zeta,
            est.T,
            est.A0,
            est.A1,
            est.A2,
        ]
        self.estimate_pub.publish(msg)

        t_msg = Float64()
        t_msg.data = est.T
        self.zv_t_pub.publish(t_msg)

    def _log_sample(self, t_motion: float, x_mm: float):
        if self.csv_writer is None or self.identifier.t0 is None:
            return
        id_t = t_motion - self.identifier.t0
        x_rel_mm = x_mm - self.identifier.x_zero
        self.csv_writer.writerow([
            'sample',
            self.get_clock().now().nanoseconds * 1.0e-9,
            t_motion,
            id_t,
            x_mm,
            x_rel_mm,
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
        ])
        if self.csv_file is not None:
            self.csv_file.flush()

    def _log_estimate(self, t_motion: float, x_mm: float, est: Estimate):
        if self.csv_writer is None:
            return
        freq_hz = est.omega_n / (2.0 * math.pi)
        self.csv_writer.writerow([
            'estimate',
            self.get_clock().now().nanoseconds * 1.0e-9,
            t_motion,
            est.t,
            x_mm,
            est.x,
            est.i1,
            est.i2,
            est.i3,
            est.cond_b,
            est.omega_n,
            freq_hz,
            est.zeta,
            est.T,
            est.A0,
            est.A1,
            est.A2,
        ])
        if self.csv_file is not None:
            self.csv_file.flush()

    def _maybe_print_not_started(self):
        now = time.monotonic()
        last = getattr(self, '_last_not_started_print', 0.0)
        if now - last < self.args.print_period:
            return
        self._last_not_started_print = now
        self.get_logger().info(
            'payload received; waiting for MOTION_START (enable motors in TRAJ)')

    def _maybe_print_waiting(self):
        now = time.monotonic()
        last = getattr(self, '_last_wait_print', 0.0)
        if now - last < self.args.print_period:
            return
        self._last_wait_print = now
        latest = self.identifier.latest_valid
        if latest is None:
            self.get_logger().info(
                'waiting: B ill-conditioned or estimate not in range yet')
        else:
            self.get_logger().info(
                f'latest valid: t={latest.t:.3f}s wn={latest.omega_n:.4f} '
                f'zeta={latest.zeta:.4f} T={latest.T:.4f}s condB={latest.cond_b:.3e}')

    def _maybe_print_estimate(self, est: Estimate):
        now = time.monotonic()
        last = getattr(self, '_last_est_print', 0.0)
        if now - last < self.args.print_period:
            return
        self._last_est_print = now

        freq_hz = est.omega_n / (2.0 * math.pi)
        amp_sum = est.A0 + est.A1 + est.A2
        self.get_logger().info(
            'ID '
            f't={est.t:7.3f}s | '
            f'x={est.x:9.3f} mm | '
            f'condB={est.cond_b:10.3e} | '
            f'wn={est.omega_n:8.4f} rad/s ({freq_hz:7.4f} Hz) | '
            f'zeta={est.zeta:7.4f} | '
            f'T={est.T:7.4f}s | '
            f'A=[{est.A0:.4f}, {est.A1:.4f}, {est.A2:.4f}] sum={amp_sum:.4f}'
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--payload-topic', default='/payload/state',
        help='Default: /payload/state')
    parser.add_argument(
        '--payload-type', default='payloadstate',
        choices=[
            'float64', 'float64multiarray', 'payloadstate',
            'pointstamped', 'posestamped',
        ],
        help='Default: payloadstate')
    parser.add_argument(
        '--payload-units', default='m', choices=['m', 'mm'],
        help='Tracker poses are meters. Default: m')
    parser.add_argument(
        '--swing-field', default='gantry_x',
        choices=[
            'gantry_z', 'gantry_x', 'gantry_y',
            'cam_x', 'cam_y', 'cam_z', 'x1', 'z1', 'x2', 'z2',
        ],
        help='PayloadState field for ID displacement. Default: gantry_x',
    )
    parser.add_argument(
        '--axis',
        default='x',
        choices=['x', 'y'],
        help='Gantry motion axis for subtracting trolley position. Default: x',
    )
    parser.add_argument(
        '--pose-x-index', type=int, default=3,
        help='Index for float64multiarray only.',
    )
    parser.add_argument('--payload-sign', type=float, default=1.0)
    parser.add_argument('--payload-offset-mm', type=float, default=0.0)
    parser.add_argument(
        '--no-zero-at-start',
        dest='zero_at_start',
        action='store_false',
        help='Do not subtract the payload position captured at MOTION_START.',
    )
    parser.add_argument('--vx-mm-s', type=float, default=60.0)
    parser.add_argument('--vy-mm-s', type=float, default=0.0)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--max-travel-mm', type=float, default=250.0)
    parser.add_argument('--a0', type=float, default=0.25)
    parser.add_argument('--cond-threshold', type=float, default=1.0e8)
    parser.add_argument('--omega-min', type=float, default=0.5)
    parser.add_argument('--omega-max', type=float, default=20.0)
    parser.add_argument('--zeta-min', type=float, default=0.0)
    parser.add_argument('--zeta-max', type=float, default=0.99)
    parser.add_argument('--print-period', type=float, default=0.25)
    parser.add_argument(
        '--estimate-topic',
        default='/adaptive_id/estimate',
        help='Float64MultiArray estimate topic.')
    parser.add_argument(
        '--zv-t-topic',
        default='/adaptive_id/zv_T',
        help='Float64 candidate shaper-period topic.')
    parser.add_argument(
        '--log-csv',
        default='',
        help='Optional CSV path for raw samples and accepted estimates.')
    parser.add_argument('--no-arm', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--start-immediately', action='store_true')
    return parser.parse_args()


def check_motion_limits(args) -> bool:
    travel_x_mm = abs(args.vx_mm_s) * args.duration
    travel_y_mm = abs(args.vy_mm_s) * args.duration
    if travel_x_mm > args.max_travel_mm:
        print(
            f'Refusing: x travel {travel_x_mm:.1f} mm > {args.max_travel_mm:.1f} mm',
            file=sys.stderr)
        return False
    if travel_y_mm > args.max_travel_mm:
        print(
            f'Refusing: y travel {travel_y_mm:.1f} mm > {args.max_travel_mm:.1f} mm',
            file=sys.stderr)
        return False
    if args.duration <= 0.0:
        print('Refusing: --duration must be positive', file=sys.stderr)
        return False
    if abs(args.vx_mm_s) < 1e-9 and abs(args.vy_mm_s) < 1e-9:
        print('Refusing: vx and vy are both zero', file=sys.stderr)
        return False
    if not (0.0 < args.a0 < 1.0):
        print('Refusing: --a0 must be in (0, 1)', file=sys.stderr)
        return False
    return True


def main():
    args = parse_args()
    if not check_motion_limits(args):
        return 1

    rclpy.init()
    node = None
    try:
        node = AdaptiveIDPlayer(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[adaptive_id_player] {exc}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
