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

Payload topics:
  /payload/pose_e_rel  Float64MultiArray encoder-relative payload position (default)
  /payload/state       PayloadState from camera tracker
  /payload/pose        legacy Float64MultiArray (camera x1/z1 per tag)

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

from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger
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


@dataclass
class IdCandidate:
    t: float
    cond_b: float
    omega_n: float = float('nan')
    zeta: float = float('nan')
    T: float = float('nan')
    valid: bool = False
    reject_reason: str = ''


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
        lowpass_hz: float = 0.0,
        local_window_s: float = 0.0,
    ):
        self.K = float(K_mm_s)
        self.A0 = float(A0)
        self.cond_threshold = float(cond_threshold)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.zeta_min = float(zeta_min)
        self.zeta_max = float(zeta_max)
        self.lowpass_hz = max(0.0, float(lowpass_hz))
        self.local_window_s = max(0.0, float(local_window_s))
        self.reset()

    def reset(self):
        self.t0: float | None = None
        self.prev_t: float | None = None
        self.prev_x: float | None = None
        self.prev_q: float | None = None
        self.x_zero = 0.0
        self.q_zero = 0.0
        self.I1 = 0.0
        self.I2 = 0.0
        self.I3 = 0.0
        self.Q1 = 0.0
        self.Q2 = 0.0
        self.Q3 = 0.0
        self.latest_valid: Estimate | None = None
        self.latest_candidate: IdCandidate | None = None
        self.x_filtered: float | None = None
        self.samples: list[tuple[float, float, float]] = []

    def start(
        self,
        t_motion: float,
        x_now_mm: float,
        q_now_mm: float = 0.0,
        zero_at_start: bool = True,
    ):
        """Reset integrators at motion start (t=0 at MOTION_START)."""
        self.reset()
        self.x_zero = float(x_now_mm) if zero_at_start else 0.0
        self.q_zero = float(q_now_mm) if zero_at_start else 0.0
        x_rel_mm = float(x_now_mm) - self.x_zero
        self.x_filtered = x_rel_mm
        q_rel_mm = float(q_now_mm) - self.q_zero
        self.t0 = float(t_motion)
        self.prev_t = float(t_motion)
        self.prev_x = x_rel_mm
        self.prev_q = q_rel_mm
        self.samples = [(0.0, x_rel_mm, q_rel_mm)]

    def update(
        self,
        t_motion: float,
        x_now_mm: float,
        q_now_mm: float | None = None,
    ) -> Estimate | None:
        if self.t0 is None:
            return None

        if self.prev_t is None or self.prev_x is None:
            self.prev_t = t_motion
            self.prev_x = x_now_mm
            return None

        t = t_motion - self.t0
        dt = t_motion - self.prev_t
        x_rel_raw_mm = float(x_now_mm) - self.x_zero
        if q_now_mm is None:
            q_rel_mm = self.K * t
        else:
            q_rel_mm = float(q_now_mm) - self.q_zero

        if dt <= 0.0 or t <= 0.0:
            return None
        if self.lowpass_hz > 0.0:
            prev_filtered = self.x_filtered if self.x_filtered is not None else x_rel_raw_mm
            alpha = 1.0 - math.exp(-2.0 * math.pi * self.lowpass_hz * dt)
            alpha = max(0.0, min(alpha, 1.0))
            x_rel_mm = prev_filtered + alpha * (x_rel_raw_mm - prev_filtered)
            self.x_filtered = x_rel_mm
        else:
            x_rel_mm = x_rel_raw_mm
            self.x_filtered = x_rel_mm

        I1_old = self.I1
        I2_old = self.I2
        Q1_old = self.Q1
        Q2_old = self.Q2
        x_avg = 0.5 * (self.prev_x + x_rel_mm)
        prev_q = self.prev_q if self.prev_q is not None else 0.0
        q_avg = 0.5 * (prev_q + q_rel_mm)

        self.I1 += x_avg * dt
        self.I2 += 0.5 * (I1_old + self.I1) * dt
        self.I3 += 0.5 * (I2_old + self.I2) * dt
        self.Q1 += q_avg * dt
        self.Q2 += 0.5 * (Q1_old + self.Q1) * dt
        self.Q3 += 0.5 * (Q2_old + self.Q2) * dt

        self.prev_t = t_motion
        self.prev_x = x_rel_mm
        self.prev_q = q_rel_mm
        self.samples.append((t, x_rel_mm, q_rel_mm))
        if self.local_window_s > 0.0:
            keep_after = max(0.0, t - self.local_window_s - 0.05)
            self.samples = [sample for sample in self.samples if sample[0] >= keep_after]
            return self._windowed_estimate(t)

        I1 = self.I1
        I2 = self.I2
        I3 = self.I3
        Q2 = self.Q2
        Q3 = self.Q3

        B = np.array(
            [
                [I1, I2 - Q2],
                [I2, I3 - Q3],
            ],
            dtype=float,
        )
        rhs = np.array([-x_rel_mm, -I1], dtype=float)

        try:
            cond_b = float(np.linalg.cond(B))
        except np.linalg.LinAlgError:
            self.latest_candidate = IdCandidate(t=t, cond_b=float('inf'), reject_reason='cond_failed')
            return None

        if not math.isfinite(cond_b) or cond_b > self.cond_threshold:
            self.latest_candidate = IdCandidate(t=t, cond_b=cond_b, reject_reason='bad_cond')
            return None

        try:
            theta = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            self.latest_candidate = IdCandidate(t=t, cond_b=cond_b, reject_reason='solve_failed')
            return None

        two_zeta_omega = float(theta[0])
        omega_sq = float(theta[1])

        if not math.isfinite(omega_sq) or omega_sq <= 0.0:
            self.latest_candidate = IdCandidate(t=t, cond_b=cond_b, reject_reason='bad_omega_sq')
            return None

        omega_n = math.sqrt(omega_sq)
        zeta = two_zeta_omega / (2.0 * omega_n)

        if not math.isfinite(omega_n) or not math.isfinite(zeta):
            self.latest_candidate = IdCandidate(
                t=t,
                cond_b=cond_b,
                omega_n=omega_n,
                zeta=zeta,
                reject_reason='bad_omega_zeta',
            )
            return None

        if not (self.omega_min <= omega_n <= self.omega_max):
            self.latest_candidate = IdCandidate(
                t=t,
                cond_b=cond_b,
                omega_n=omega_n,
                zeta=zeta,
                reject_reason='omega_gate',
            )
            return None

        if not (self.zeta_min <= zeta <= self.zeta_max):
            self.latest_candidate = IdCandidate(
                t=t,
                cond_b=cond_b,
                omega_n=omega_n,
                zeta=zeta,
                reject_reason='zeta_gate',
            )
            return None

        shaper = self.compute_shaper(zeta, omega_n)
        if shaper is None:
            self.latest_candidate = IdCandidate(
                t=t,
                cond_b=cond_b,
                omega_n=omega_n,
                zeta=zeta,
                reject_reason='shaper_failed',
            )
            return None

        T, A0, A1, A2 = shaper
        self.latest_candidate = IdCandidate(
            t=t,
            cond_b=cond_b,
            omega_n=omega_n,
            zeta=zeta,
            T=T,
            valid=True,
        )
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

    def _windowed_estimate(self, t_now: float) -> Estimate | None:
        """Estimate over a local window while fitting local initial velocity.

        Eq. (21) assumes the payload starts the ID interval with zero velocity.
        A sliding window generally violates that, so this local form solves for
        [2*zeta*omega, omega**2, xdot_window_start] with least squares.
        """
        window_start = max(0.0, t_now - self.local_window_s)
        window = [sample for sample in self.samples if sample[0] >= window_start]
        if len(window) < 8:
            self.latest_candidate = IdCandidate(t=t_now, cond_b=float('inf'), reject_reason='local_short')
            return None

        t0, x0, q0 = window[0]
        times = np.asarray([sample[0] - t0 for sample in window], dtype=float)
        x_vals = np.asarray([sample[1] for sample in window], dtype=float)
        q_vals = np.asarray([sample[2] for sample in window], dtype=float)
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(x_vals)) and np.all(np.isfinite(q_vals))):
            self.latest_candidate = IdCandidate(t=t_now, cond_b=float('inf'), reject_reason='local_bad_samples')
            return None

        span = float(times[-1] - times[0])
        if span < 0.20:
            self.latest_candidate = IdCandidate(t=t_now, cond_b=float('inf'), reject_reason='local_short_span')
            return None

        x_rel = x_vals - x0
        q_rel = q_vals - q0
        swing0 = x0 - q0

        I1 = np.zeros_like(times)
        Q1 = np.zeros_like(times)
        I2 = np.zeros_like(times)
        Q2 = np.zeros_like(times)
        I3 = np.zeros_like(times)
        Q3 = np.zeros_like(times)
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if dt <= 0.0:
                continue
            I1[i] = I1[i - 1] + 0.5 * (x_rel[i - 1] + x_rel[i]) * dt
            Q1[i] = Q1[i - 1] + 0.5 * (q_rel[i - 1] + q_rel[i]) * dt
            I2[i] = I2[i - 1] + 0.5 * (I1[i - 1] + I1[i]) * dt
            Q2[i] = Q2[i - 1] + 0.5 * (Q1[i - 1] + Q1[i]) * dt
            I3[i] = I3[i - 1] + 0.5 * (I2[i - 1] + I2[i]) * dt
            Q3[i] = Q3[i - 1] + 0.5 * (Q2[i - 1] + Q2[i]) * dt

        rows: list[list[float]] = []
        rhs: list[float] = []
        for i in range(1, len(times)):
            tau = float(times[i])
            if tau <= 0.0:
                continue
            spring_i2 = float(I2[i] - Q2[i] + 0.5 * swing0 * tau * tau)
            spring_i3 = float(I3[i] - Q3[i] + swing0 * tau * tau * tau / 6.0)
            rows.append([float(I1[i]), spring_i2, -tau])
            rhs.append(float(-x_rel[i]))
            rows.append([float(I2[i]), spring_i3, -0.5 * tau * tau])
            rhs.append(float(-I1[i]))

        if len(rows) < 6:
            self.latest_candidate = IdCandidate(t=t_now, cond_b=float('inf'), reject_reason='local_short_rows')
            return None

        A = np.asarray(rows, dtype=float)
        b = np.asarray(rhs, dtype=float)
        scales = np.linalg.norm(A, axis=0)
        if np.any(scales <= 1.0e-12) or not np.all(np.isfinite(scales)):
            self.latest_candidate = IdCandidate(t=t_now, cond_b=float('inf'), reject_reason='local_bad_scale')
            return None

        A_scaled = A / scales
        try:
            cond_b = float(np.linalg.cond(A_scaled))
        except np.linalg.LinAlgError:
            self.latest_candidate = IdCandidate(t=t_now, cond_b=float('inf'), reject_reason='local_cond_failed')
            return None
        if not math.isfinite(cond_b) or cond_b > self.cond_threshold:
            self.latest_candidate = IdCandidate(t=t_now, cond_b=cond_b, reject_reason='local_bad_cond')
            return None

        try:
            theta_scaled, _, rank, _ = np.linalg.lstsq(A_scaled, b, rcond=None)
        except np.linalg.LinAlgError:
            self.latest_candidate = IdCandidate(t=t_now, cond_b=cond_b, reject_reason='local_solve_failed')
            return None
        if rank < 3:
            self.latest_candidate = IdCandidate(t=t_now, cond_b=cond_b, reject_reason='local_rank')
            return None

        theta = theta_scaled / scales
        two_zeta_omega = float(theta[0])
        omega_sq = float(theta[1])
        if not math.isfinite(omega_sq) or omega_sq <= 0.0:
            self.latest_candidate = IdCandidate(t=t_now, cond_b=cond_b, reject_reason='local_bad_omega_sq')
            return None

        omega_n = math.sqrt(omega_sq)
        zeta = two_zeta_omega / (2.0 * omega_n)
        if not math.isfinite(omega_n) or not math.isfinite(zeta):
            self.latest_candidate = IdCandidate(
                t=t_now,
                cond_b=cond_b,
                omega_n=omega_n,
                zeta=zeta,
                reject_reason='local_bad_omega_zeta',
            )
            return None
        if not (self.omega_min <= omega_n <= self.omega_max):
            self.latest_candidate = IdCandidate(
                t=t_now,
                cond_b=cond_b,
                omega_n=omega_n,
                zeta=zeta,
                reject_reason='local_omega_gate',
            )
            return None
        if not (self.zeta_min <= zeta <= self.zeta_max):
            self.latest_candidate = IdCandidate(
                t=t_now,
                cond_b=cond_b,
                omega_n=omega_n,
                zeta=zeta,
                reject_reason='local_zeta_gate',
            )
            return None

        shaper = self.compute_shaper(zeta, omega_n)
        if shaper is None:
            self.latest_candidate = IdCandidate(
                t=t_now,
                cond_b=cond_b,
                omega_n=omega_n,
                zeta=zeta,
                reject_reason='local_shaper_failed',
            )
            return None

        T, A0, A1, A2 = shaper
        self.latest_candidate = IdCandidate(
            t=t_now,
            cond_b=cond_b,
            omega_n=omega_n,
            zeta=zeta,
            T=T,
            valid=True,
        )
        est = Estimate(
            t=t_now,
            x=float(x_rel[-1]),
            i1=float(I1[-1]),
            i2=float(I2[-1]),
            i3=float(I3[-1]),
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
        if zeta >= 1.0:
            return None

        # The gantry experiments in the ACC paper set damping to zero for the
        # shaper. Noisy online integral estimates can briefly report small
        # negative damping; allow callers to gate that via zeta_min, then use
        # the physically meaningful zero-damping shaper.
        zeta_shape = max(0.0, zeta)
        wd_factor = math.sqrt(max(1.0 - zeta_shape**2, 1.0e-12))
        T = math.pi / (omega_n * wd_factor)

        exp1 = math.exp(math.pi * zeta_shape / wd_factor)
        exp2 = math.exp(2.0 * math.pi * zeta_shape / wd_factor)

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
        self.latest_swing_x_mm: float | None = None
        self.latest_cart_q_mm: float | None = None
        self.latest_motion_t: float | None = None
        self.csv_file = None
        self.csv_writer: csv.writer | None = None
        self._zv_t_filtered: float | None = None
        self._zv_valid_count = 0
        self._zv_param_pending = False
        self._zv_param_last_send = 0.0
        self._zv_param_cli = None
        self._ringdown_samples: list[tuple[float, float]] = []
        self._ringdown_last_est_t = 0.0
        self._auto_enable_timer = None

        id_velocity = args.vx_mm_s if args.axis == 'x' else args.vy_mm_s

        self.identifier = AdaptiveIdentifier(
            K_mm_s=id_velocity,
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
        self.enable_cli = self.create_client(Trigger, '/gantry/enable')
        self.estimate_pub = self.create_publisher(
            Float64MultiArray, args.estimate_topic, 10)
        self.zv_t_pub = self.create_publisher(Float64, args.zv_t_topic, 10)
        if args.apply_zv_param:
            self._zv_param_cli = self.create_client(
                SetParameters, f'{args.zv_param_node}/set_parameters')

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
        elif args.payload_type == 'encoderrel':
            field = 'x_rel_m' if args.axis == 'x' else 'y_rel_m'
            self.get_logger().info(f'Encoder field: {field}')
        else:
            self.get_logger().info(f'Pose x index:  {args.pose_x_index}')
        self.get_logger().info(
            f'ID profile: vx={args.vx_mm_s:.1f} mm/s, '
            f'vy={args.vy_mm_s:.1f} mm/s, duration={args.duration:.1f} s')
        self.get_logger().info(f'ID method: {args.id_method}')
        if args.id_method in ('integral', 'auto'):
            window_text = 'unlimited' if args.integral_window <= 0.0 else f'{args.integral_window:.2f}s'
            self.get_logger().info(f'Integral ID window: {window_text}')
        self.get_logger().info(
            f'ID axis: {args.axis}  K={id_velocity:.1f} mm/s')
        self.get_logger().info(f'Estimate topic: {args.estimate_topic}')
        self.get_logger().info(f'ZV T topic:      {args.zv_t_topic}')
        if args.apply_zv_param:
            self.get_logger().warn(
                f'Adaptive ZV apply enabled: {args.zv_param_node}.zv_T '
                f'clamp=[{args.zv_t_min:.3f}, {args.zv_t_max:.3f}] '
                f'alpha={args.zv_t_alpha:.2f}')
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
        elif msg_type == 'encoderrel':
            self.create_subscription(
                Float64MultiArray, topic, self._payload_encoder_rel_cb, qos)
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
                'encoderrel, payloadstate, pointstamped, posestamped')

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
            self._begin_id(
                self.latest_motion_t,
                self.latest_payload_x_mm,
                self.latest_cart_q_mm,
            )
        else:
            self.get_logger().warn(
                'MOTION_START: waiting for first payload sample to start ID')

    def _begin_id(self, t_motion: float, x_mm: float, q_mm: float | None = None):
        self.pending_motion_start = False
        self._ringdown_samples.clear()
        self._ringdown_last_est_t = 0.0
        self.identifier.start(
            t_motion,
            x_mm,
            q_now_mm=0.0 if q_mm is None else q_mm,
            zero_at_start=self.args.zero_at_start)
        zero_text = f' zero={self.identifier.x_zero:.2f} mm'
        self.get_logger().info(
            f'MOTION_START: ID started at t={t_motion:.3f}s '
            f'x={x_mm:.2f} mm q={0.0 if q_mm is None else q_mm:.2f} mm{zero_text}')

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
        if self.latest_gantry_state is not None and bool(self.latest_gantry_state.enabled):
            delay = max(0.0, float(self.args.auto_enable_delay))
            self._auto_enable_timer = self.create_timer(delay, self._auto_enable_timer_cb)
        else:
            self._start_traj_if_motors_enabled()

        if self.args.start_immediately:
            self._start_id_immediately()

    def _auto_enable_timer_cb(self):
        if self._auto_enable_timer is not None:
            self._auto_enable_timer.cancel()
            self._auto_enable_timer = None
        self._start_traj_if_motors_enabled()

    def _start_traj_if_motors_enabled(self):
        if self.args.no_start_if_enabled:
            return
        if self.latest_gantry_state is None or not bool(self.latest_gantry_state.enabled):
            self.get_logger().info(
                'Motors are off — press Enable to start TRAJ playback')
            return
        if not self.enable_cli.service_is_ready():
            if not self.enable_cli.wait_for_service(timeout_sec=0.5):
                self.get_logger().warn(
                    '/gantry/enable not available; press Enable to start playback')
                return
        future = self.enable_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_enable_done)
        self.get_logger().info(
            'Motors already enabled — requested /gantry/enable to start TRAJ playback')

    def _on_enable_done(self, future):
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warn(f'/gantry/enable failed: {exc}')
            return
        if result is None:
            self.get_logger().warn('/gantry/enable returned no result')
            return
        if result.success:
            self.get_logger().info(f'/gantry/enable: {result.message}')
        else:
            self.get_logger().warn(f'/gantry/enable rejected: {result.message}')

    def _start_id_immediately(self):
        if self.latest_payload_x_mm is None:
            self.get_logger().warn('--start-immediately: no payload sample yet')
            return
        t = self.latest_motion_t if self.latest_motion_t is not None else 0.0
        self.motion_started = True
        self._begin_id(t, self.latest_payload_x_mm, self.latest_cart_q_mm)
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

    def _payload_encoder_rel_cb(self, msg: Float64MultiArray):
        # /payload/pose_e_rel:
        # [time, pitch_deg, roll_deg, x_rel_m, y_rel_m, z_rel_m, vx, vy, vz]
        index = 3 if self.args.axis == 'x' else 4
        if len(msg.data) <= index:
            self.get_logger().warn('Encoder relative array too short')
            return
        swing_m = float(msg.data[index])
        cart_m = self._current_cart_axis_m()
        if cart_m is None:
            # Integral ID needs cart position history. Keep relative swing for
            # ringdown, but use q=0 until gantry state arrives.
            cart_m = 0.0
        abs_payload_m = cart_m + swing_m
        self._handle_payload_measurement(
            x_id_mm=abs_payload_m * 1000.0,
            t_motion=float(msg.data[0]),
            swing_x_mm=swing_m * 1000.0,
            q_mm=cart_m * 1000.0,
        )

    def _current_cart_axis_m(self) -> float | None:
        if self.latest_gantry_state is None:
            return None
        return (
            float(self.latest_gantry_state.x)
            if self.args.axis == 'x'
            else float(self.latest_gantry_state.y)
        )

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
        q_m = self._current_cart_axis_m()
        q_mm = None if q_m is None else q_m * 1000.0
        self._handle_payload_measurement(
            x_id_mm=x_mm,
            t_motion=t_motion,
            swing_x_mm=x_mm,
            q_mm=q_mm,
        )

    def _handle_payload_measurement(
        self,
        x_id_mm: float,
        t_motion: float | None,
        swing_x_mm: float | None = None,
        q_mm: float | None = None,
    ):
        x_mm = float(x_id_mm)
        swing_mm = x_mm if swing_x_mm is None else float(swing_x_mm)

        self.latest_payload_x_mm = x_mm
        self.latest_swing_x_mm = swing_mm
        self.latest_cart_q_mm = q_mm
        if t_motion is not None:
            self.latest_motion_t = float(t_motion)

        if self.pending_motion_start and t_motion is not None:
            self._begin_id(t_motion, x_mm, q_mm)
            return

        if not self.motion_started:
            self._maybe_print_not_started()
            return

        if t_motion is None:
            self.get_logger().warn_throttle(
                5.0, 'No motion_time in payload; use /payload/pose or /payload/state')
            return

        if self.identifier.t0 is None:
            self._begin_id(t_motion, x_mm, q_mm)
            return

        self._log_sample(t_motion, x_mm)
        est = None
        id_t = t_motion - self.identifier.t0
        integral_active = (
            self.args.integral_window <= 0.0
            or id_t <= self.args.integral_window
        )
        if self.args.id_method in ('integral', 'auto') and integral_active:
            est = self.identifier.update(t_motion, x_mm, q_mm)
        if est is None and self.args.id_method in ('ringdown', 'auto'):
            est = self._update_ringdown_estimate(t_motion, swing_mm)
        if est is None:
            self._maybe_print_waiting()
            return

        self._publish_estimate(est)
        self._log_estimate(t_motion, x_mm, est)
        self._maybe_print_estimate(est)

    def _update_ringdown_estimate(self, t_motion: float, x_mm: float) -> Estimate | None:
        if self.identifier.t0 is None:
            return None
        id_t = t_motion - self.identifier.t0
        x_rel_mm = x_mm - self.identifier.x_zero
        start_t = self.args.duration + self.args.ringdown_delay
        end_t = start_t + self.args.ringdown_window
        if id_t < start_t:
            return None
        if id_t > end_t:
            return None

        self._ringdown_samples.append((id_t, x_rel_mm))
        max_samples = int(max(50, self.args.ringdown_window * 250.0))
        if len(self._ringdown_samples) > max_samples:
            self._ringdown_samples = self._ringdown_samples[-max_samples:]
        if id_t - self._ringdown_last_est_t < self.args.ringdown_update_period:
            return None
        self._ringdown_last_est_t = id_t

        return self._estimate_ringdown(id_t)

    def _estimate_ringdown(self, id_t: float) -> Estimate | None:
        samples = self._ringdown_samples
        if len(samples) < 20:
            return None

        smooth = []
        radius = max(0, int(self.args.ringdown_smooth_samples))
        for i, (t, _) in enumerate(samples):
            lo = max(0, i - radius)
            hi = min(len(samples), i + radius + 1)
            x = sum(samples[j][1] for j in range(lo, hi)) / float(hi - lo)
            smooth.append((t, x))

        extrema: list[tuple[float, float]] = []
        for i in range(1, len(smooth) - 1):
            t, x = smooth[i]
            xp = smooth[i - 1][1]
            xn = smooth[i + 1][1]
            if abs(x) < self.args.ringdown_min_peak_mm:
                continue
            if (x >= xp and x > xn) or (x <= xp and x < xn):
                if extrema and t - extrema[-1][0] < self.args.ringdown_min_peak_dt:
                    if abs(x) > abs(extrema[-1][1]):
                        extrema[-1] = (t, x)
                else:
                    extrema.append((t, x))

        if len(extrema) < 4:
            return None

        half_periods = [
            extrema[i + 1][0] - extrema[i][0]
            for i in range(len(extrema) - 1)
            if extrema[i + 1][0] > extrema[i][0]
        ]
        half_periods = [
            dt for dt in half_periods
            if self.args.ringdown_half_period_min <= dt <= self.args.ringdown_half_period_max
        ]
        if len(half_periods) < 3:
            return None

        half_period = float(np.median(np.array(half_periods, dtype=float)))
        T_d = 2.0 * half_period
        omega_d = 2.0 * math.pi / T_d

        deltas = []
        for i in range(len(extrema) - 2):
            a0 = abs(extrema[i][1])
            a1 = abs(extrema[i + 2][1])
            if a0 > self.args.ringdown_min_peak_mm and a1 > 1.0e-6 and a0 > a1:
                deltas.append(math.log(a0 / a1))
        if deltas:
            delta = float(np.median(np.array(deltas, dtype=float)))
            zeta = delta / math.sqrt(4.0 * math.pi * math.pi + delta * delta)
            zeta = max(self.args.zeta_min, min(zeta, self.args.zeta_max))
        else:
            zeta = max(self.args.zeta_min, min(0.02, self.args.zeta_max))

        omega_n = omega_d / math.sqrt(max(1.0 - zeta * zeta, 1.0e-12))
        if not (self.args.omega_min <= omega_n <= self.args.omega_max):
            return None

        shaper = self.identifier.compute_shaper(zeta, omega_n)
        if shaper is None:
            return None
        T, A0, A1, A2 = shaper
        latest_x = samples[-1][1]
        est = Estimate(
            t=id_t,
            x=latest_x,
            i1=self.identifier.I1,
            i2=self.identifier.I2,
            i3=self.identifier.I3,
            cond_b=0.0,
            omega_n=omega_n,
            zeta=zeta,
            T=T,
            A0=A0,
            A1=A1,
            A2=A2,
        )
        self.identifier.latest_valid = est
        return est

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
        self._maybe_apply_zv_t(est.T)

    def _maybe_apply_zv_t(self, t_est: float):
        if not self.args.apply_zv_param or self._zv_param_cli is None:
            return
        if not math.isfinite(t_est):
            return

        t_clamped = max(self.args.zv_t_min, min(float(t_est), self.args.zv_t_max))
        if self._zv_t_filtered is None:
            self._zv_t_filtered = t_clamped
        else:
            a = self.args.zv_t_alpha
            self._zv_t_filtered = a * t_clamped + (1.0 - a) * self._zv_t_filtered

        self._zv_valid_count += 1
        if self._zv_valid_count < self.args.zv_min_valid:
            return

        now = time.monotonic()
        if self._zv_param_pending:
            return
        if now - self._zv_param_last_send < self.args.zv_apply_period:
            return
        if not self._zv_param_cli.service_is_ready():
            if not self._zv_param_cli.wait_for_service(timeout_sec=0.0):
                return

        param = Parameter()
        param.name = 'zv_T'
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE,
            double_value=float(self._zv_t_filtered),
        )
        req = SetParameters.Request()
        req.parameters = [param]
        future = self._zv_param_cli.call_async(req)
        self._zv_param_pending = True
        self._zv_param_last_send = now
        future.add_done_callback(self._on_zv_param_done)

    def _on_zv_param_done(self, future):
        self._zv_param_pending = False
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warn(f'zv_T parameter update failed: {exc}')
            return
        if result is None or not result.results:
            self.get_logger().warn('zv_T parameter update returned no result')
            return
        if not result.results[0].successful:
            reason = result.results[0].reason or 'unknown reason'
            self.get_logger().warn(f'zv_T parameter rejected: {reason}')

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
        '--payload-topic', default='/payload/pose_e_rel',
        help='Default: /payload/pose_e_rel')
    parser.add_argument(
        '--payload-type', default='encoderrel',
        choices=[
            'float64', 'float64multiarray', 'encoderrel', 'payloadstate',
            'pointstamped', 'posestamped',
        ],
        help='Default: encoderrel (/payload/pose_e_rel)')
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
        help='ID/motion axis. encoderrel uses x_rel_m for x, y_rel_m for y. Default: x',
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
        '--integral-window',
        type=float,
        default=2.5,
        help='Seconds after MOTION_START to run integral ID. <=0 disables the limit.',
    )
    parser.add_argument(
        '--id-method',
        default='auto',
        choices=['auto', 'integral', 'ringdown'],
        help='System ID method. auto uses integral first, then ringdown after stop.',
    )
    parser.add_argument('--ringdown-delay', type=float, default=0.15)
    parser.add_argument('--ringdown-window', type=float, default=8.0)
    parser.add_argument('--ringdown-update-period', type=float, default=0.25)
    parser.add_argument('--ringdown-min-peak-mm', type=float, default=5.0)
    parser.add_argument('--ringdown-min-peak-dt', type=float, default=0.15)
    parser.add_argument('--ringdown-half-period-min', type=float, default=0.35)
    parser.add_argument('--ringdown-half-period-max', type=float, default=2.0)
    parser.add_argument('--ringdown-smooth-samples', type=int, default=2)
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
    parser.add_argument(
        '--apply-zv-param',
        action='store_true',
        help='Apply filtered estimated T to /gantry_controller zv_T.')
    parser.add_argument(
        '--zv-param-node',
        default='/gantry_controller',
        help='Controller node for SetParameters. Default: /gantry_controller')
    parser.add_argument('--zv-t-min', type=float, default=0.2)
    parser.add_argument('--zv-t-max', type=float, default=3.0)
    parser.add_argument('--zv-t-alpha', type=float, default=0.20)
    parser.add_argument('--zv-min-valid', type=int, default=5)
    parser.add_argument('--zv-apply-period', type=float, default=0.5)
    parser.add_argument('--no-arm', action='store_true')
    parser.add_argument(
        '--no-start-if-enabled',
        action='store_true',
        help='Do not call /gantry/enable after loading TRAJ when motors are already on.',
    )
    parser.add_argument(
        '--auto-enable-delay',
        type=float,
        default=0.25,
        help='Delay before auto-calling /gantry/enable when motors are already on.',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--start-immediately', action='store_true')
    return parser.parse_args()


def check_motion_limits(args) -> bool:
    travel_x_mm = abs(args.vx_mm_s) * args.duration
    travel_y_mm = abs(args.vy_mm_s) * args.duration
    id_velocity = args.vx_mm_s if args.axis == 'x' else args.vy_mm_s
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
    if abs(id_velocity) < 1e-9:
        print(
            f'Refusing: --axis {args.axis} selected but that axis velocity is zero',
            file=sys.stderr)
        return False
    if not (0.0 < args.a0 < 1.0):
        print('Refusing: --a0 must be in (0, 1)', file=sys.stderr)
        return False
    if args.zv_t_min <= 0.0 or args.zv_t_max <= args.zv_t_min:
        print('Refusing: invalid --zv-t-min/--zv-t-max', file=sys.stderr)
        return False
    if not (0.0 < args.zv_t_alpha <= 1.0):
        print('Refusing: --zv-t-alpha must be in (0, 1]', file=sys.stderr)
        return False
    if args.zv_min_valid < 1:
        print('Refusing: --zv-min-valid must be >= 1', file=sys.stderr)
        return False
    if args.zv_apply_period <= 0.0:
        print('Refusing: --zv-apply-period must be positive', file=sys.stderr)
        return False
    if args.auto_enable_delay < 0.0:
        print('Refusing: --auto-enable-delay must be non-negative', file=sys.stderr)
        return False
    if args.ringdown_delay < 0.0:
        print('Refusing: --ringdown-delay must be non-negative', file=sys.stderr)
        return False
    if args.ringdown_window <= 0.0:
        print('Refusing: --ringdown-window must be positive', file=sys.stderr)
        return False
    if args.ringdown_update_period <= 0.0:
        print('Refusing: --ringdown-update-period must be positive', file=sys.stderr)
        return False
    if args.ringdown_min_peak_mm <= 0.0:
        print('Refusing: --ringdown-min-peak-mm must be positive', file=sys.stderr)
        return False
    if args.ringdown_min_peak_dt <= 0.0:
        print('Refusing: --ringdown-min-peak-dt must be positive', file=sys.stderr)
        return False
    if args.ringdown_half_period_max <= args.ringdown_half_period_min:
        print('Refusing: invalid ringdown half-period bounds', file=sys.stderr)
        return False
    if args.ringdown_smooth_samples < 0:
        print('Refusing: --ringdown-smooth-samples must be >= 0', file=sys.stderr)
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
