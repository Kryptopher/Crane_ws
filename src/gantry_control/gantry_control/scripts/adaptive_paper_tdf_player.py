#!/usr/bin/env python3
"""
adaptive_paper_tdf_player.py

Paper-style adaptive time-delay-filter player for one-axis gantry tests.

This follows the ACC 2024 reference-shaping structure more directly than the
stop-only adaptive player:

  1. Start the target move at A0 * vmax to excite the payload.
  2. Estimate the pendulum switch time T online during that early motion.
  3. Lock a TDF schedule and command the staircase velocity profile:

       A0*vmax, (A0+A1)*vmax, vmax, (1-A0)*vmax,
       (1-A0-A1)*vmax, 0

     with switches at T, 2T, tf, tf+T, tf+2T where tf = distance/vmax.

The "robust" profile is a model-based ZVD baseline. It computes T and the
three ZVD weights from a configured rope length and does not use online ID.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
import sympy as sy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import SetMode

from adaptive_id_player import AdaptiveIdentifier, Estimate


@dataclass
class LockedSchedule:
    source: str
    locked_at: float
    id_time: float
    id_T: float
    T: float
    A0: float
    A1: float
    A2: float
    raw_id_zeta: float
    id_zeta: float
    shaper_zeta: float
    cond_b: float


class AdaptivePaperTdfPlayer(Node):
    def __init__(self, args):
        super().__init__('adaptive_paper_tdf_player')
        self.args = args

        self.latest_gantry_state: GantryState | None = None
        self.latest_payload_abs_mm: float | None = None
        self.latest_swing_mm: float | None = None
        self.latest_cart_q_mm: float | None = None
        self.latest_payload_time: float | None = None
        self.latest_payload_wall: float | None = None

        self.motion_started = False
        self.start_requested = False
        self.wall_start: float | None = None
        self.start_cart_q_mm: float | None = None
        self.final_zero_wall: float | None = None
        self.done = False
        self.aborted = False

        self.schedule: LockedSchedule | None = None
        self.valid_estimates: list[Estimate] = []
        self.id_report_estimates: list[Estimate] = []
        self.best_estimate: Estimate | None = None
        self.residual_samples: list[tuple[float, float]] = []
        self.csv_file = None
        self.csv_writer: csv.writer | None = None
        self._log_flush_count = 0
        self._last_wait_ready_print = 0.0
        self._last_id_status_print = 0.0
        self._missed_first_switch_warned = False

        self.direction = 1.0 if args.vmax_mm_s >= 0.0 else -1.0
        self.vmax_abs_mm_s = abs(float(args.vmax_mm_s))
        self.tf = args.target_distance_mm / self.vmax_abs_mm_s
        self.initial_velocity_mm_s = self.direction * args.a0 * self.vmax_abs_mm_s
        if args.profile == 'robust':
            self._configure_robust_schedule()

        self.identifier = AdaptiveIdentifier(
            K_mm_s=self.initial_velocity_mm_s,
            A0=abs(self.initial_velocity_mm_s) / self.vmax_abs_mm_s,
            cond_threshold=args.cond_threshold,
            omega_min=args.omega_min,
            omega_max=args.omega_max,
            zeta_min=args.id_zeta_min,
            zeta_max=args.id_zeta_max,
        )

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.traj_pub = self.create_publisher(TrajCmd, '/traj_cmd', qos)
        self.estimate_pub = self.create_publisher(Float64MultiArray, args.estimate_topic, 10)
        self.enable_cli = self.create_client(Trigger, '/gantry/enable')
        self.mode_cli = self.create_client(SetMode, '/gantry/set_mode')

        self.create_subscription(GantryState, '/gantry/state', self._gantry_state_cb, 10)
        self.create_subscription(TrajCmd, '/traj_cmd', self._traj_cmd_cb, qos)
        self.create_subscription(
            Float64MultiArray, args.payload_topic, self._payload_cb, qos_profile_sensor_data)

        if args.log_csv:
            self._open_csv_log(args.log_csv)
        if not args.no_arm:
            self._arm_traj_mode()

        self.stream_timer = self.create_timer(1.0 / args.stream_rate_hz, self._stream_timer_cb)
        self.start_timer = self.create_timer(0.5, self._start_timer_cb)

        self.get_logger().info('adaptive_paper_tdf_player started')
        self.get_logger().info(
            f'Profile={args.profile} axis={args.axis} target={args.target_distance_mm:.1f} mm '
            f'vmax={args.vmax_mm_s:.1f} mm/s A0={args.a0:.3f} '
            f'initial_v={self.initial_velocity_mm_s:.1f} mm/s tf={self.tf:.3f}s')
        if args.profile in (
            'adaptive',
            'colleague-moving',
            'colleague-velocity-exact',
            'colleague-paper-closed',
        ):
            self.get_logger().info(
                f'Accept: condB<{args.accept_cond:.1f}, {args.accept_valid_count} valid estimates, '
                f'T in [{args.zv_t_min:.2f}, {args.zv_t_max:.2f}]s; '
                f'id_lock_mode={args.id_lock_mode}; fallback at {args.estimate_deadline:.2f}s '
                f'{"enabled" if args.allow_fallback else "disabled"}')
            self.get_logger().info(
                f'ID damping gate: zeta in [{args.id_zeta_min:.3f}, {args.id_zeta_max:.3f}], '
                f'shaper damping clamp: [{args.zeta_min:.3f}, {args.zeta_max:.3f}]')
            if args.profile in (
                'colleague-moving',
                'colleague-velocity-exact',
                'colleague-paper-closed',
            ):
                self.get_logger().info(
                    f'{args.profile} IS: tau={args.tau:.3f}s, initial speed K*vmax, '
                    'then switches at tau+T, tau+2T, tf, tf+tau+T, tf+tau+2T')
        elif args.profile == 'robust':
            sched = self.schedule
            if sched is not None:
                omega = math.sqrt(args.gravity_m_s2 / args.robust_rope_length_m)
                self.get_logger().info(
                    f'Robust ZVD baseline: L={args.robust_rope_length_m:.3f}m '
                    f'omega_n={omega:.4f}rad/s T={sched.T:.4f}s '
                    f'A=[{sched.A0:.4f},{sched.A1:.4f},{sched.A2:.4f}]')
        else:
            self.get_logger().info('Pulse baseline: command vmax until tf, then zero.')

    def destroy_node(self):
        try:
            if rclpy.ok():
                self._publish_stream(0.0, 0.0)
        except Exception:
            pass
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
            'wall_time_sec',
            'move_time_sec',
            'cmd_vx_mm_s',
            'cmd_vy_mm_s',
            'payload_abs_mm',
            'swing_mm',
            'cart_q_mm',
            'traveled_mm',
            'cond_b',
            'omega_n_rad_s',
            'id_zeta',
            'T_sec',
            'A0',
            'A1',
            'A2',
            'schedule_source',
            'schedule_locked_at_s',
            'schedule_id_time_s',
            'schedule_id_T_sec',
            'schedule_T_sec',
        ])
        self.csv_file.flush()
        self.get_logger().info(f'CSV log: {path}')

    def _arm_traj_mode(self):
        if not self.mode_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/gantry/set_mode not available')
        req = SetMode.Request()
        req.mode = 'TRAJ'
        req.csv_path = ''
        future = self.mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not result.success:
            msg = result.message if result is not None else 'no response'
            raise RuntimeError(f'Failed to set TRAJ mode: {msg}')
        self.get_logger().info('Gantry set to TRAJ mode')

    def _start_timer_cb(self):
        if not self._preflight_ready():
            now = time.monotonic()
            if now - self._last_wait_ready_print > 1.0:
                self._last_wait_ready_print = now
                self.get_logger().info(
                    'Waiting for fresh /payload/pose_e_rel and /gantry/state before start')
            return

        try:
            self.start_timer.cancel()
        except Exception:
            pass
        if self.args.no_auto_enable:
            self.get_logger().info('Press Enable to start TRAJ realtime STREAM')
            return
        if not self.enable_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('/gantry/enable unavailable; press Enable manually')
            self.start_requested = True
            return
        self.start_requested = True
        future = self.enable_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_enable_done)
        self.get_logger().info('Requested /gantry/enable to start TRAJ realtime STREAM')

    def _preflight_ready(self) -> bool:
        if self.latest_gantry_state is None:
            return False
        if self.latest_payload_abs_mm is None or self.latest_payload_wall is None:
            return False
        return time.monotonic() - self.latest_payload_wall <= self.args.payload_fresh_timeout

    def _on_enable_done(self, future):
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warn(f'/gantry/enable failed: {exc}')
            return
        if result is not None:
            self.get_logger().info(f'/gantry/enable: {result.message}')

    def _gantry_state_cb(self, msg: GantryState):
        self.latest_gantry_state = msg

    def _traj_cmd_cb(self, msg: TrajCmd):
        if msg.command != TrajCmd.MOTION_START or self.motion_started:
            return
        self.motion_started = True
        self.wall_start = time.monotonic()
        if self.latest_payload_abs_mm is not None and self.latest_payload_time is not None:
            self._begin_id()
        self.get_logger().info('MOTION_START: paper TDF move started')

    def _payload_cb(self, msg: Float64MultiArray):
        index = 3 if self.args.axis == 'x' else 4
        if len(msg.data) <= index:
            self.get_logger().warn('Encoder relative array too short')
            return

        swing_m = float(msg.data[index])
        cart_m = self._current_cart_axis_m()
        if cart_m is None:
            cart_m = 0.0

        self.latest_payload_time = float(msg.data[0])
        self.latest_payload_wall = time.monotonic()
        self.latest_swing_mm = swing_m * 1000.0
        self.latest_cart_q_mm = cart_m * 1000.0
        self.latest_payload_abs_mm = (cart_m + swing_m) * 1000.0

        if self.final_zero_wall is not None:
            residual_t = self.latest_payload_wall - self.final_zero_wall
            if residual_t >= 0.0:
                self.residual_samples.append((residual_t, self.latest_swing_mm))

        if self.motion_started and self.identifier.t0 is None:
            self._begin_id()
        if not self.motion_started or self.identifier.t0 is None:
            return

        est = self.identifier.update(
            self.latest_payload_time,
            self.latest_payload_abs_mm,
            self.latest_cart_q_mm,
        )
        if est is not None:
            self._publish_estimate(est)
            self._record_id_report_estimate(est)
            self._maybe_lock_from_estimate(est)

    def _current_cart_axis_m(self) -> float | None:
        if self.latest_gantry_state is None:
            return None
        return (
            float(self.latest_gantry_state.x)
            if self.args.axis == 'x'
            else float(self.latest_gantry_state.y)
        )

    def _begin_id(self):
        if self.latest_payload_time is None or self.latest_payload_abs_mm is None:
            return
        q = 0.0 if self.latest_cart_q_mm is None else self.latest_cart_q_mm
        if self.start_cart_q_mm is None:
            self.start_cart_q_mm = q
        self.identifier.start(
            self.latest_payload_time,
            self.latest_payload_abs_mm,
            q_now_mm=q,
            zero_at_start=True,
        )
        self.get_logger().info(
            f'ID started: x={self.latest_payload_abs_mm:.2f} mm q={q:.2f} mm')

    def _publish_estimate(self, est: Estimate):
        msg = Float64MultiArray()
        msg.data = [
            est.t,
            est.x,
            est.cond_b,
            est.omega_n,
            est.omega_n / (2.0 * math.pi),
            est.zeta,
            est.T,
            est.A0,
            est.A1,
            est.A2,
        ]
        self.estimate_pub.publish(msg)

    def _schedule_zeta_for_estimate(self, est: Estimate) -> float:
        return est.zeta

    def _record_id_report_estimate(self, est: Estimate):
        if self.wall_start is None:
            return
        if est.t < self.args.min_id_duration or est.t > self.args.id_period:
            return
        if self.args.profile == 'adaptive' and not self._estimate_passes_accept_gates(est):
            return
        if self._is_colleague_profile() and not self._estimate_passes_colleague_id_gates(est):
            return
        if self.args.profile not in (
            'adaptive',
            'colleague-moving',
            'colleague-velocity-exact',
            'colleague-paper-closed',
        ):
            return
        self.id_report_estimates.append(est)

    def _configure_robust_schedule(self):
        omega_n = math.sqrt(self.args.gravity_m_s2 / self.args.robust_rope_length_m)
        T = self.args.robust_t_scale * math.pi / omega_n
        A0, A1, A2 = self._standard_zvd_amplitudes(self.args.robust_zeta)
        self.initial_velocity_mm_s = self.direction * A0 * self.vmax_abs_mm_s
        self.schedule = LockedSchedule(
            source=f'robust_zvd_L{self.args.robust_rope_length_m:.3f}m',
            locked_at=0.0,
            id_time=float('nan'),
            id_T=float('nan'),
            T=T,
            A0=A0,
            A1=A1,
            A2=A2,
            raw_id_zeta=self.args.robust_zeta,
            id_zeta=self.args.robust_zeta,
            shaper_zeta=self.args.robust_zeta,
            cond_b=float('nan'),
        )

    def _standard_zvd_amplitudes(self, zeta: float) -> tuple[float, float, float]:
        z = max(0.0, min(float(zeta), 0.99))
        wd_factor = math.sqrt(max(1.0 - z * z, 1.0e-12))
        decay = math.exp(-math.pi * z / wd_factor)
        denom = 1.0 + 2.0 * decay + decay * decay
        return 1.0 / denom, 2.0 * decay / denom, decay * decay / denom

    def _maybe_lock_from_estimate(self, est: Estimate):
        if self.aborted:
            return
        if self.schedule is not None:
            return
        if self.wall_start is None:
            return
        move_t = time.monotonic() - self.wall_start
        if self.args.profile not in (
            'adaptive',
            'colleague-moving',
            'colleague-velocity-exact',
            'colleague-paper-closed',
        ):
            return
        if move_t < self.args.min_id_duration:
            return
        if self.args.profile == 'adaptive' and not self._estimate_passes_accept_gates(est):
            return
        if self._is_colleague_profile() and not self._estimate_passes_colleague_id_gates(est):
            return

        self.valid_estimates.append(est)
        if self.best_estimate is None or est.cond_b < self.best_estimate.cond_b:
            self.best_estimate = est
            if self.args.id_lock_mode == 'best-cond' or self._is_colleague_profile():
                self.get_logger().info(
                    f'new lowest-cond ID candidate: t={est.t:.3f}s T={est.T:.4f}s '
                    f'zeta={est.zeta:.4f} condB={est.cond_b:.3e}')
        if self._is_colleague_profile():
            return
        if self.args.id_lock_mode == 'best-cond':
            lock_time = max(self.args.min_id_duration, est.T - self.args.switch_margin)
            if move_t >= lock_time:
                best = self.best_estimate
                if best is not None:
                    self.get_logger().info(
                        f'Locking best estimate before first switch: '
                        f'move_t={move_t:.3f}s lock_time={lock_time:.3f}s '
                        f'best_t={best.t:.3f}s T={best.T:.4f}s condB={best.cond_b:.3e}')
                    self._lock_schedule(
                        'adaptive_best_cond',
                        move_t,
                        best.T,
                        best.zeta,
                        best.cond_b,
                    )
            return

        max_count = max(self.args.accept_valid_count, self.args.stability_count)
        self.valid_estimates = self.valid_estimates[-max_count:]
        if len(self.valid_estimates) < self.args.accept_valid_count:
            return

        recent = self.valid_estimates[-self.args.stability_count:]
        if len(recent) >= 2:
            t_values = [e.T for e in recent]
            if max(t_values) - min(t_values) > self.args.stability_tol:
                return

        self._lock_schedule('adaptive', move_t, est.T, est.zeta, est.cond_b)

    def _lock_schedule(self, source: str, move_t: float, T: float, id_zeta: float, cond_b: float):
        shaper_zeta = max(self.args.zeta_min, min(id_zeta, self.args.zeta_max))
        A0, A1, A2 = self._shaper_amplitudes(shaper_zeta)
        self.schedule = LockedSchedule(
            source=source,
            locked_at=move_t,
            id_time=move_t,
            id_T=T,
            T=T,
            A0=A0,
            A1=A1,
            A2=A2,
            raw_id_zeta=id_zeta,
            id_zeta=id_zeta,
            shaper_zeta=shaper_zeta,
            cond_b=cond_b,
        )
        switches = [T, 2.0 * T, self.tf, self.tf + T, self.tf + 2.0 * T]
        levels = [
            A0 * self.vmax_abs_mm_s,
            (A0 + A1) * self.vmax_abs_mm_s,
            self.vmax_abs_mm_s,
            (1.0 - A0) * self.vmax_abs_mm_s,
            max(0.0, 1.0 - A0 - A1) * self.vmax_abs_mm_s,
        ]
        signed_levels = [self.direction * v for v in levels]
        self.get_logger().info(
            f'LOCKED {source} TDF: t={move_t:.3f}s T={T:.4f}s '
            f'id_zeta={id_zeta:.4f} shaper_zeta={shaper_zeta:.4f} '
            f'condB={cond_b:.3e} A=[{A0:.4f},{A1:.4f},{A2:.4f}]')
        self.get_logger().info(
            'Paper TDF switches: '
            + ', '.join(f'{s:.3f}s' for s in switches)
            + ' | velocities: '
            + ', '.join(f'{v:.1f}' for v in signed_levels)
            + ' mm/s')
        if move_t > T and not self._missed_first_switch_warned:
            self._missed_first_switch_warned = True
            self.get_logger().warn(
                f'Schedule locked after first switch time: move_t={move_t:.3f}s > T={T:.3f}s')

    def _lock_colleague_schedule(self, source: str, move_t: float, est: Estimate):
        raw_id_zeta = est.zeta
        id_zeta = self._schedule_zeta_for_estimate(est)
        shaper_zeta = max(self.args.zeta_min, min(id_zeta, self.args.zeta_max))
        if self.args.profile == 'colleague-velocity-exact':
            solved = self._colleague_velocity_exact_amplitudes(est.omega_n, shaper_zeta)
            solve_name = 'colleague velocity-exact'
        elif self.args.profile == 'colleague-paper-closed':
            solved = self._colleague_paper_closed_form_amplitudes(est.omega_n)
            solve_name = 'colleague paper closed-form'
        else:
            solved = self._colleague_amplitudes(est.omega_n, shaper_zeta)
            solve_name = 'colleague'
        if solved is None:
            self.get_logger().error(
                f'{solve_name} A/T solve failed for omega_n={est.omega_n:.4f} '
                f'zeta={shaper_zeta:.4f}')
            self._abort_without_estimate(move_t, reason=f'{solve_name} A/T solve failed')
            return
        A1, T = solved
        A0 = self.args.a0
        A2 = 1.0 - A0 - A1
        if not (self.args.zv_t_min <= T <= self.args.zv_t_max):
            self.get_logger().error(f'{solve_name} solved T={T:.4f}s outside gate')
            self._abort_without_estimate(
                move_t,
                reason=(
                    f'{solve_name} solved T={T:.4f}s outside '
                    f'[{self.args.zv_t_min:.4f}, {self.args.zv_t_max:.4f}]s gate'
                ),
            )
            return
        if not (0.0 <= A1 <= 1.0 - A0 and A2 >= 0.0):
            self.get_logger().error(
                f'{solve_name} solved invalid amplitudes A=[{A0:.4f},{A1:.4f},{A2:.4f}]')
            self._abort_without_estimate(
                move_t,
                reason=f'{solve_name} solved invalid amplitudes A=[{A0:.4f},{A1:.4f},{A2:.4f}]',
            )
            return
        if self.tf <= self.args.tau + 2.0 * T:
            self.get_logger().error(
                f'Invalid moving pulse: tf={self.tf:.3f}s <= tau+2T={self.args.tau + 2.0*T:.3f}s')
            self._abort_without_estimate(
                move_t,
                reason=f'invalid moving pulse tf={self.tf:.3f}s <= tau+2T={self.args.tau + 2.0*T:.3f}s',
            )
            return

        self.schedule = LockedSchedule(
            source=source,
            locked_at=move_t,
            id_time=est.t,
            id_T=est.T,
            T=T,
            A0=A0,
            A1=A1,
            A2=A2,
            raw_id_zeta=raw_id_zeta,
            id_zeta=id_zeta,
            shaper_zeta=shaper_zeta,
            cond_b=est.cond_b,
        )
        switches = [
            self.args.tau + T,
            self.args.tau + 2.0 * T,
            self.tf,
            self.tf + self.args.tau + T,
            self.tf + self.args.tau + 2.0 * T,
        ]
        levels = [
            A0 * self.vmax_abs_mm_s,
            (A0 + A1) * self.vmax_abs_mm_s,
            self.vmax_abs_mm_s,
            (1.0 - A0) * self.vmax_abs_mm_s,
            max(0.0, 1.0 - A0 - A1) * self.vmax_abs_mm_s,
        ]
        signed_levels = [self.direction * v for v in levels]
        self.get_logger().info(
            f'LOCKED {self.args.profile} IS: t={move_t:.3f}s tau={self.args.tau:.3f}s '
            f'T={T:.4f}s id_zeta={id_zeta:.4f} shaper_zeta={shaper_zeta:.4f} '
            f'raw_id_zeta={raw_id_zeta:.4f} omega_n={est.omega_n:.4f} condB={est.cond_b:.3e} '
            f'A=[{A0:.4f},{A1:.4f},{A2:.4f}]')
        self.get_logger().info(
            f'{self.args.profile} switches: '
            + ', '.join(f'{s:.3f}s' for s in switches)
            + ' | velocities: '
            + ', '.join(f'{v:.1f}' for v in signed_levels)
            + ' mm/s')

    def _colleague_amplitudes(self, omega_n: float, zeta: float) -> tuple[float, float] | None:
        A_sym, T_sym = sy.symbols('A,T')
        K = self.args.a0
        tau = self.args.tau
        wd_factor = math.sqrt(max(1.0 - zeta * zeta, 1.0e-12))
        real = (
            K
            + A_sym * sy.exp(zeta * omega_n * (tau + T_sym))
            * sy.cos(omega_n * (tau + T_sym) * wd_factor)
            + (1.0 - K - A_sym) * sy.exp(zeta * omega_n * (tau + 2.0 * T_sym))
            * sy.cos(omega_n * (tau + 2.0 * T_sym) * wd_factor)
        )
        imag = (
            A_sym * sy.exp(zeta * omega_n * (tau + T_sym))
            * sy.sin(omega_n * (tau + T_sym) * wd_factor)
            + (1.0 - K - A_sym) * sy.exp(zeta * omega_n * (tau + 2.0 * T_sym))
            * sy.sin(omega_n * (tau + 2.0 * T_sym) * wd_factor)
        )
        T_approx = math.pi / omega_n
        for i in range(100):
            try:
                A_sol, T_sol = sy.nsolve(
                    [real, imag],
                    [A_sym, T_sym],
                    [0.0, (1.2 + 0.1 * i) * T_approx],
                    verify=False,
                )
                A_ret = float(A_sol)
                T_ret = float(T_sol)
            except Exception:
                continue
            if math.isfinite(A_ret) and math.isfinite(T_ret):
                if A_ret > 0.0 and A_ret < (1.0 - K) and T_ret > 0.4 * T_approx:
                    return A_ret, T_ret
        return None

    def _colleague_velocity_exact_amplitudes(
        self,
        omega_n: float,
        zeta: float,
    ) -> tuple[float, float] | None:
        A_sym, T_sym = sy.symbols('A,T')
        K = self.args.a0
        tau = self.args.tau
        tf = self.tf
        wd_factor = math.sqrt(max(1.0 - zeta * zeta, 1.0e-12))

        # Velocity-edge equation:
        # K + A e^-s(tau+T) + B e^-s(tau+2T)
        # - K e^-s(tf) - A e^-s(tf+tau+T) - B e^-s(tf+tau+2T) = 0
        # with B = 1-K-A. This solves the moving velocity profile directly
        # instead of applying the paper's position-step equation.
        terms = [
            (K, 0.0),
            (A_sym, tau + T_sym),
            (1.0 - K - A_sym, tau + 2.0 * T_sym),
            (-K, tf),
            (-A_sym, tf + tau + T_sym),
            (-(1.0 - K - A_sym), tf + tau + 2.0 * T_sym),
        ]
        real = 0
        imag = 0
        for coeff, delay in terms:
            exp_term = sy.exp(zeta * omega_n * delay)
            angle = omega_n * delay * wd_factor
            real += coeff * exp_term * sy.cos(angle)
            imag += coeff * exp_term * sy.sin(angle)

        T_approx = math.pi / omega_n
        guesses = []
        for a_guess in (0.05, 0.2, 0.4, 0.6):
            for scale in [0.5 + 0.05 * i for i in range(50)]:
                guesses.append((a_guess, scale * T_approx))

        best: tuple[float, float] | None = None
        for a_guess, t_guess in guesses:
            try:
                A_sol, T_sol = sy.nsolve(
                    [real, imag],
                    [A_sym, T_sym],
                    [a_guess, t_guess],
                    verify=False,
                    tol=1.0e-12,
                    maxsteps=80,
                )
                A_ret = float(A_sol)
                T_ret = float(T_sol)
            except Exception:
                continue
            if not (math.isfinite(A_ret) and math.isfinite(T_ret)):
                continue
            if A_ret <= 0.0 or A_ret >= (1.0 - K) or T_ret <= 0.0:
                continue
            if T_ret <= 0.25 * T_approx:
                continue
            if best is None or T_ret < best[1]:
                best = (A_ret, T_ret)
        return best

    def _colleague_paper_closed_form_amplitudes(self, omega_n: float) -> tuple[float, float] | None:
        K = self.args.a0
        tau = self.args.tau
        phi = omega_n * tau
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        if abs(sin_phi) < 1.0e-9:
            return None

        beta = sy.symbols('beta')
        cubic = (
            (-K * sin_phi) * beta**3
            + (K - 1.0 + 3.0 * K * cos_phi) * beta**2
            + (3.0 * K * sin_phi) * beta
            + (K - K * cos_phi - 1.0)
        )
        try:
            roots = sy.nroots(cubic, n=30, maxsteps=100)
        except Exception:
            return None

        candidates: list[tuple[float, float]] = []
        for root in roots:
            root_c = complex(root)
            if abs(root_c.imag) > 1.0e-7:
                continue
            beta_real = float(root_c.real)
            base_T = 2.0 * math.atan(beta_real) / omega_n
            for k in range(-5, 6):
                T = base_T + 2.0 * math.pi * k / omega_n
                if T <= 0.0:
                    continue
                denom = (
                    math.cos(omega_n * (tau + T))
                    - math.cos(omega_n * (tau + 2.0 * T))
                )
                if abs(denom) < 1.0e-9:
                    continue
                A = -(
                    K
                    - math.cos(omega_n * (tau + 2.0 * T)) * (K - 1.0)
                ) / denom
                A2 = 1.0 - K - A
                if not (math.isfinite(A) and math.isfinite(T)):
                    continue
                if A <= 0.0 or A >= 1.0 - K or A2 < 0.0:
                    continue
                candidates.append((A, T))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1])
        return candidates[0]

    def _is_colleague_profile(self) -> bool:
        return self.args.profile in (
            'colleague-moving',
            'colleague-velocity-exact',
            'colleague-paper-closed',
        )

    def _stream_timer_cb(self):
        if self.done:
            return
        if not self.start_requested and not self.motion_started:
            return
        now = time.monotonic()
        if self.wall_start is None:
            self._publish_stream(0.0, 0.0)
            return

        move_t = now - self.wall_start
        if self.aborted:
            self._publish_stream(0.0, 0.0)
            if now - self.final_zero_wall >= self.args.residual_window:
                self.done = True
                self._print_final_report()
            return

        if (
            self._is_colleague_profile()
            and self.schedule is None
            and move_t >= self.args.tau
        ):
            source, selected = self._select_colleague_lock_estimate(move_t)
            latest = self.identifier.latest_valid
            if selected is not None:
                self.get_logger().info(
                    f'Using {source} estimate at tau: '
                    f'id_t={selected.t:.3f}s T={selected.T:.4f}s omega={selected.omega_n:.4f} '
                    f'zeta={selected.zeta:.4f} '
                    f'condB={selected.cond_b:.3e}')
                self._lock_colleague_schedule(source, move_t, selected)
            elif latest is not None and self._estimate_passes_colleague_id_gates(latest):
                self.get_logger().info(
                    f'Using latest valid estimate at tau: '
                    f'id_t={latest.t:.3f}s omega={latest.omega_n:.4f} '
                    f'zeta={latest.zeta:.4f} condB={latest.cond_b:.3e}')
                source = self.args.profile.replace('-', '_') + '_latest'
                self._lock_colleague_schedule(source, move_t, latest)
            elif self.args.allow_fallback:
                if move_t >= self.args.estimate_deadline:
                    self.get_logger().warn(
                    f'No colleague estimate by deadline; fallback not implemented for {self.args.profile}')
                    self._abort_without_estimate(move_t)
            else:
                if move_t >= self.args.estimate_deadline:
                    self._abort_without_estimate(move_t)

        if self.aborted:
            self._publish_stream(0.0, 0.0)
            return

        if (
            self.args.profile == 'adaptive'
            and self.schedule is None
            and move_t >= self.args.estimate_deadline
        ):
            best = self.best_estimate
            latest = self.identifier.latest_valid
            if self.args.id_lock_mode == 'best-cond' and best is not None:
                self.get_logger().info(
                    f'Using best conditioning estimate at deadline: '
                    f'id_t={best.t:.3f}s T={best.T:.4f}s condB={best.cond_b:.3e}')
                self._lock_schedule('adaptive_best_cond', move_t, best.T, best.zeta, best.cond_b)
            elif latest is not None and self._estimate_passes_accept_gates(latest):
                self.get_logger().info(
                    f'Using latest valid estimate at deadline: '
                    f'id_t={latest.t:.3f}s T={latest.T:.4f}s condB={latest.cond_b:.3e}')
                self._lock_schedule('adaptive_deadline', move_t, latest.T, latest.zeta, latest.cond_b)
            elif self.args.allow_fallback:
                self._lock_schedule(
                    'fallback',
                    move_t,
                    self.args.fallback_t,
                    0.0,
                    float('nan'),
                )
            else:
                self._abort_without_estimate(move_t)

        if self.aborted:
            self._publish_stream(0.0, 0.0)
            return

        vx_axis = self._axis_velocity_for_time(move_t)
        vx, vy = (vx_axis, 0.0) if self.args.axis == 'x' else (0.0, vx_axis)
        self._publish_stream(vx, vy)
        self._log_row(move_t, vx, vy)

        end_t = self._end_time()
        if move_t >= end_t and self.final_zero_wall is None:
            self.final_zero_wall = now
            self._publish_stream(0.0, 0.0)
            self.get_logger().info(
                f'Paper TDF command is zero; collecting residual swing for '
                f'{self.args.residual_window:.1f}s')
            return
        if self.final_zero_wall is not None:
            self._publish_stream(0.0, 0.0)
            if now - self.final_zero_wall >= self.args.residual_window:
                self.done = True
                self._print_final_report()
        elif (
            self.args.profile in (
                'adaptive',
                'colleague-moving',
                'colleague-velocity-exact',
                'colleague-paper-closed',
            )
            and self.schedule is None
            and now - self._last_id_status_print >= self.args.print_period
        ):
            self._last_id_status_print = now
            latest = self.identifier.latest_valid
            if latest is None:
                self.get_logger().info('waiting for online ID estimate before first switch')
            else:
                self.get_logger().info(
                    f'latest ID before lock: t={latest.t:.3f}s T={latest.T:.4f}s '
                    f'zeta={latest.zeta:.4f} condB={latest.cond_b:.3e}')
            if self.best_estimate is not None:
                best = self.best_estimate
                self.get_logger().info(
                    f'best ID so far: t={best.t:.3f}s T={best.T:.4f}s '
                    f'omega={best.omega_n:.4f} zeta={best.zeta:.4f} '
                    f'condB={best.cond_b:.3e}')

    def _select_colleague_lock_estimate(self, move_t: float) -> tuple[str, Estimate | None]:
        source_base = self.args.profile.replace('-', '_')
        if self.args.profile != 'colleague-paper-closed':
            if self.best_estimate is not None:
                return source_base + '_best_cond', self.best_estimate
            return source_base + '_latest', None

        candidates = [
            est
            for est in self.valid_estimates
            if est.t <= move_t and self._estimate_passes_colleague_id_gates(est)
        ]
        if not candidates:
            return source_base + '_median_T', None
        t_values = sorted(est.T for est in candidates)
        mid = len(t_values) // 2
        median_T = (
            t_values[mid]
            if len(t_values) % 2 == 1
            else 0.5 * (t_values[mid - 1] + t_values[mid])
        )
        selected = min(
            candidates,
            key=lambda est: (abs(est.T - median_T), -est.t, est.cond_b),
        )
        self.get_logger().info(
            f'IS2 median-T selection: median_T={median_T:.4f}s from '
            f'{len(candidates)} valid estimates; selected_t={selected.t:.3f}s '
            f'selected_T={selected.T:.4f}s')
        return source_base + '_median_T', selected

    def _abort_without_estimate(self, move_t: float, reason: str | None = None):
        if self.aborted:
            return
        self.aborted = True
        self.final_zero_wall = time.monotonic()
        self._publish_stream(0.0, 0.0)
        reason_text = (
            f'No adaptive estimate by {move_t:.3f}s and fallback disabled'
            if reason is None
            else reason
        )
        self.get_logger().error(
            f'{reason_text}; '
            f'stopping STREAM and collecting residual for {self.args.residual_window:.1f}s')

    def _axis_velocity_for_time(self, move_t: float) -> float:
        if self.args.profile == 'pulse':
            return self.direction * self.vmax_abs_mm_s if move_t < self.tf else 0.0

        sched = self.schedule
        if sched is None:
            return self.initial_velocity_mm_s

        T = sched.T
        A0, A1 = sched.A0, sched.A1
        if self._is_colleague_profile():
            tau = self.args.tau
            if move_t < tau + T:
                gain = A0
            elif move_t < tau + 2.0 * T:
                gain = A0 + A1
            elif move_t < self.tf:
                gain = 1.0
            elif move_t < self.tf + tau + T:
                gain = 1.0 - A0
            elif move_t < self.tf + tau + 2.0 * T:
                gain = 1.0 - A0 - A1
            else:
                gain = 0.0
            return self.direction * self.vmax_abs_mm_s * max(0.0, gain)

        if move_t < T:
            gain = A0
        elif move_t < 2.0 * T:
            gain = A0 + A1
        elif move_t < self.tf:
            gain = 1.0
        elif move_t < self.tf + T:
            gain = 1.0 - A0
        elif move_t < self.tf + 2.0 * T:
            gain = 1.0 - A0 - A1
        else:
            gain = 0.0
        return self.direction * self.vmax_abs_mm_s * max(0.0, gain)

    def _estimate_passes_accept_gates(self, est: Estimate) -> bool:
        if est.cond_b > self.args.accept_cond:
            return False
        if not (self.args.zv_t_min <= est.T <= self.args.zv_t_max):
            return False
        return self.args.id_zeta_min <= est.zeta <= self.args.id_zeta_max

    def _estimate_passes_colleague_id_gates(self, est: Estimate) -> bool:
        if est.cond_b > self.args.accept_cond:
            return False
        if not (self.args.omega_min <= est.omega_n <= self.args.omega_max):
            return False
        return self.args.id_zeta_min <= est.zeta <= self.args.id_zeta_max

    def _end_time(self) -> float:
        if self.args.profile == 'pulse':
            return self.tf
        if self._is_colleague_profile():
            T = self.args.fallback_t if self.schedule is None else self.schedule.T
            return self.tf + self.args.tau + 2.0 * T
        T = self.args.fallback_t if self.schedule is None else self.schedule.T
        return self.tf + 2.0 * T

    def _shaper_amplitudes(self, zeta: float) -> tuple[float, float, float]:
        z = max(0.0, min(float(zeta), 0.99))
        wd_factor = math.sqrt(max(1.0 - z * z, 1.0e-12))
        exp1 = math.exp(math.pi * z / wd_factor)
        exp2 = math.exp(2.0 * math.pi * z / wd_factor)
        A0 = self.args.a0
        A1 = (A0 + (1.0 - A0) * exp2) / (exp1 + exp2)
        A2 = 1.0 - A0 - A1
        return A0, A1, A2

    def _traveled_target_distance_mm(self) -> float:
        if self.start_cart_q_mm is None or self.latest_cart_q_mm is None:
            return 0.0
        return self.direction * (self.latest_cart_q_mm - self.start_cart_q_mm)

    def _print_final_report(self):
        traveled = self._traveled_target_distance_mm()
        target_error = traveled - self.args.target_distance_mm
        samples = [x for _, x in self.residual_samples]
        if samples:
            residual_max = max(abs(x) for x in samples)
            residual_p2p = max(samples) - min(samples)
            residual_rms = math.sqrt(sum(x * x for x in samples) / len(samples))
            sample_text = (
                f'residual swing over {self.args.residual_window:.1f}s: '
                f'p2p={residual_p2p:.2f} mm max_abs={residual_max:.2f} mm '
                f'rms={residual_rms:.2f} mm samples={len(samples)}'
            )
        else:
            sample_text = 'residual swing: no post-stop payload samples'

        if self.schedule is None:
            estimate_text = 'estimate: no schedule locked'
        else:
            estimate_text = (
                f'estimate: {self.schedule.source} T={self.schedule.T:.4f}s '
                f'zeta={self.schedule.id_zeta:.4f} condB={self.schedule.cond_b:.3e}'
            )
            if self._is_colleague_profile():
                estimate_text += f' raw_id_zeta={self.schedule.raw_id_zeta:.4f}'
            if self._is_colleague_profile() and math.isfinite(self.schedule.id_time):
                estimate_text += (
                    f' chosen_id_t={self.schedule.id_time:.3f}s'
                    f' chosen_id_T={self.schedule.id_T:.4f}s'
                )

        id_text = ''
        if self._is_colleague_profile() and self.id_report_estimates:
            first = self.id_report_estimates[0]
            last = self.id_report_estimates[-1]
            if self.schedule is not None and math.isfinite(self.schedule.id_time):
                chosen_text = (
                    f'chosen t={self.schedule.id_time:.3f}s '
                    f'ID_T={self.schedule.id_T:.4f}s schedule_T={self.schedule.T:.4f}s'
                )
            else:
                chosen_text = 'chosen none'
            id_text = (
                f' | ID period {self.args.id_period:.1f}s: first t={first.t:.3f}s T={first.T:.4f}s; '
                f'last t={last.t:.3f}s T={last.T:.4f}s; {chosen_text}'
            )

        if self.args.profile == 'pulse':
            status = 'Pulse baseline move complete'
        else:
            status = 'Paper TDF move aborted' if self.aborted else 'Paper TDF move complete'
        self.get_logger().info(
            f'{status} | '
            f'travel: target={self.args.target_distance_mm:.1f} mm '
            f'actual={traveled:.1f} mm error={target_error:+.1f} mm | '
            f'{sample_text} | {estimate_text}{id_text}')

    def _publish_stream(self, vx_mm_s: float, vy_mm_s: float):
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = TrajCmd.STREAM
        msg.vx_mm_s = float(vx_mm_s)
        msg.vy_mm_s = float(vy_mm_s)
        if self.latest_gantry_state is not None:
            msg.x = float(self.latest_gantry_state.x)
            msg.y = float(self.latest_gantry_state.y)
        self.traj_pub.publish(msg)

    def _log_row(self, move_t: float, vx: float, vy: float):
        if self.csv_writer is None:
            return
        latest = self.identifier.latest_valid
        sched = self.schedule
        source = 'pulse' if self.args.profile == 'pulse' else ('' if sched is None else sched.source)
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds * 1.0e-9,
            move_t,
            vx,
            vy,
            '' if self.latest_payload_abs_mm is None else self.latest_payload_abs_mm,
            '' if self.latest_swing_mm is None else self.latest_swing_mm,
            '' if self.latest_cart_q_mm is None else self.latest_cart_q_mm,
            self._traveled_target_distance_mm(),
            '' if latest is None else latest.cond_b,
            '' if latest is None else latest.omega_n,
            '' if latest is None else latest.zeta,
            '' if latest is None else latest.T,
            '' if sched is None else sched.A0,
            '' if sched is None else sched.A1,
            '' if sched is None else sched.A2,
            source,
            '' if sched is None else sched.locked_at,
            '' if sched is None else sched.id_time,
            '' if sched is None else sched.id_T,
            '' if sched is None else sched.T,
        ])
        if self.csv_file is not None:
            self._log_flush_count += 1
            if self._log_flush_count % 25 == 0:
                self.csv_file.flush()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis', default='x', choices=['x', 'y'])
    parser.add_argument(
        '--profile',
        default='adaptive',
        choices=[
            'adaptive',
            'pulse',
            'colleague-moving',
            'colleague-velocity-exact',
            'colleague-paper-closed',
            'robust',
        ])
    parser.add_argument('--payload-topic', default='/payload/pose_e_rel')
    parser.add_argument('--target-distance-mm', type=float, default=600.0)
    parser.add_argument('--vmax-mm-s', type=float, default=200.0)
    parser.add_argument('--a0', type=float, default=0.5)
    parser.add_argument(
        '--tau',
        type=float,
        default=0.75,
        help='For colleague profiles: timing offset used in the closed-form IS profile.')
    parser.add_argument(
        '--id-lock-mode',
        choices=['first', 'best-cond'],
        default='first',
        help='first locks as soon as acceptance gates pass; best-cond keeps lowest condB until switch/estimate deadline.')
    parser.add_argument('--min-id-duration', type=float, default=0.5)
    parser.add_argument(
        '--id-period',
        type=float,
        default=4.0,
        help='Duration of valid ID estimates to include in first/last ID reporting.')
    parser.add_argument(
        '--switch-margin',
        type=float,
        default=0.05,
        help='In best-cond mode, lock by estimated T minus this margin so the first switch is not missed.')
    parser.add_argument(
        '--estimate-deadline',
        type=float,
        default=0.9,
        help='Latest time to lock an adaptive estimate before fallback/abort.')
    parser.add_argument('--max-travel-mm', type=float, default=650.0)
    parser.add_argument('--stream-rate-hz', type=float, default=100.0)
    parser.add_argument('--print-period', type=float, default=0.25)
    parser.add_argument('--payload-fresh-timeout', type=float, default=0.25)
    parser.add_argument('--cond-threshold', type=float, default=1.0e8)
    parser.add_argument('--accept-cond', type=float, default=1000.0)
    parser.add_argument('--accept-valid-count', type=int, default=1)
    parser.add_argument('--stability-count', type=int, default=1)
    parser.add_argument('--stability-tol', type=float, default=0.08)
    parser.add_argument('--omega-min', type=float, default=0.5)
    parser.add_argument('--omega-max', type=float, default=20.0)
    parser.add_argument('--id-zeta-min', type=float, default=0.0)
    parser.add_argument('--id-zeta-max', type=float, default=0.99)
    parser.add_argument('--zeta-min', type=float, default=0.0)
    parser.add_argument('--zeta-max', type=float, default=0.05)
    parser.add_argument('--zv-t-min', type=float, default=0.2)
    parser.add_argument('--zv-t-max', type=float, default=1.25)
    parser.add_argument('--fallback-t', type=float, default=1.04)
    parser.add_argument(
        '--robust-rope-length-m',
        type=float,
        default=1.20,
        help='For --profile robust: nominal rope length used for model-based ZVD.')
    parser.add_argument(
        '--robust-zeta',
        type=float,
        default=0.0,
        help='For --profile robust: damping ratio used for model-based ZVD weights.')
    parser.add_argument(
        '--robust-t-scale',
        type=float,
        default=1.0,
        help='For --profile robust: multiplier on T=pi/sqrt(g/L).')
    parser.add_argument(
        '--gravity-m-s2',
        type=float,
        default=9.80665,
        help='Gravity used by --profile robust.')
    parser.set_defaults(allow_fallback=True)
    parser.add_argument(
        '--no-fallback',
        dest='allow_fallback',
        action='store_false',
        help='Require an adaptive estimate; abort the STREAM if none is available by estimate-deadline.')
    parser.add_argument('--residual-window', type=float, default=2.0)
    parser.add_argument('--log-csv', default='')
    parser.add_argument('--no-arm', action='store_true')
    parser.add_argument('--no-auto-enable', action='store_true')
    parser.add_argument('--estimate-topic', default='/adaptive_paper_tdf/estimate')
    return parser.parse_args()


def check_args(args) -> bool:
    if args.target_distance_mm <= 0.0:
        print('Refusing: --target-distance-mm must be positive', file=sys.stderr)
        return False
    if abs(args.vmax_mm_s) < 1.0e-9:
        print('Refusing: --vmax-mm-s must be nonzero', file=sys.stderr)
        return False
    if not (0.0 < args.a0 < 1.0):
        print('Refusing: --a0 must be in (0, 1)', file=sys.stderr)
        return False
    if args.tau <= 0.0:
        print('Refusing: --tau must be positive', file=sys.stderr)
        return False
    if args.id_lock_mode == 'best-cond' and args.estimate_deadline <= args.min_id_duration:
        print(
            'Refusing: best-cond mode needs --estimate-deadline > --min-id-duration',
            file=sys.stderr,
        )
        return False
    if args.switch_margin < 0.0:
        print('Refusing: --switch-margin must be non-negative', file=sys.stderr)
        return False
    if args.stream_rate_hz <= 0.0:
        print('Refusing: --stream-rate-hz must be positive', file=sys.stderr)
        return False
    if args.print_period <= 0.0:
        print('Refusing: --print-period must be positive', file=sys.stderr)
        return False
    if args.payload_fresh_timeout <= 0.0:
        print('Refusing: --payload-fresh-timeout must be positive', file=sys.stderr)
        return False
    if args.min_id_duration < 0.0:
        print('Refusing: --min-id-duration must be non-negative', file=sys.stderr)
        return False
    if args.estimate_deadline <= 0.0:
        print('Refusing: --estimate-deadline must be positive', file=sys.stderr)
        return False
    if args.id_period <= 0.0:
        print('Refusing: --id-period must be positive', file=sys.stderr)
        return False
    if args.accept_valid_count < 1 or args.stability_count < 1:
        print('Refusing: accept/stability counts must be >= 1', file=sys.stderr)
        return False
    if args.stability_tol <= 0.0:
        print('Refusing: --stability-tol must be positive', file=sys.stderr)
        return False
    if args.zv_t_min <= 0.0 or args.zv_t_max <= args.zv_t_min:
        print('Refusing: invalid --zv-t-min/--zv-t-max', file=sys.stderr)
        return False
    if args.id_zeta_max <= args.id_zeta_min:
        print('Refusing: invalid --id-zeta-min/--id-zeta-max', file=sys.stderr)
        return False
    if args.zeta_min < 0.0 or args.zeta_max < args.zeta_min:
        print('Refusing: invalid --zeta-min/--zeta-max', file=sys.stderr)
        return False
    if args.fallback_t <= 0.0:
        print('Refusing: --fallback-t must be positive', file=sys.stderr)
        return False
    if args.robust_rope_length_m <= 0.0:
        print('Refusing: --robust-rope-length-m must be positive', file=sys.stderr)
        return False
    if not (0.0 <= args.robust_zeta < 1.0):
        print('Refusing: --robust-zeta must be in [0, 1)', file=sys.stderr)
        return False
    if args.robust_t_scale <= 0.0:
        print('Refusing: --robust-t-scale must be positive', file=sys.stderr)
        return False
    if args.gravity_m_s2 <= 0.0:
        print('Refusing: --gravity-m-s2 must be positive', file=sys.stderr)
        return False
    if args.residual_window <= 0.0:
        print('Refusing: --residual-window must be positive', file=sys.stderr)
        return False
    if args.target_distance_mm > args.max_travel_mm:
        print(
            f'Refusing: target travel {args.target_distance_mm:.1f} mm > '
            f'{args.max_travel_mm:.1f} mm.',
            file=sys.stderr,
        )
        return False

    tf = args.target_distance_mm / abs(args.vmax_mm_s)
    if args.profile == 'robust':
        omega_n = math.sqrt(args.gravity_m_s2 / args.robust_rope_length_m)
        T_guard = args.robust_t_scale * math.pi / omega_n
    else:
        T_guard = min(args.zv_t_max, max(args.fallback_t, 1.0))
    if args.profile in (
        'colleague-moving',
        'colleague-velocity-exact',
        'colleague-paper-closed',
    ):
        # The middle full-speed interval requires tf > tau + 2T.
        # Use the configured T upper gate as a conservative guard.
        if tf <= args.tau + 2.0 * args.zv_t_max:
            print(
                f'Refusing: {args.profile} needs tf > tau + 2*Tmax. '
                f'Here tf={tf:.3f}s and tau+2*Tmax={args.tau + 2.0 * args.zv_t_max:.3f}s. '
                'Use lower --vmax-mm-s, smaller --tau/--zv-t-max, or larger distance.',
                file=sys.stderr,
            )
            return False
    elif tf <= 2.0 * T_guard:
        print(
            f'Refusing: paper TDF needs tf=distance/vmax > 2T. '
            f'Here tf={tf:.3f}s and guard 2T={2.0 * T_guard:.3f}s. '
            'Use lower --vmax-mm-s or larger --target-distance-mm.',
            file=sys.stderr,
        )
        return False
    return True


def main():
    args = parse_args()
    if not check_args(args):
        return 1
    rclpy.init()
    node = None
    try:
        node = AdaptivePaperTdfPlayer(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[adaptive_paper_tdf_player] {exc}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
