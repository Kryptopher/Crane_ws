#!/usr/bin/env python3
"""
payload_stabilizer.py

Active residual-swing damping tool for the gantry crane.

This is intentionally separate from the input-shaping experiment players. It is
meant to run after any move, for example after a dashboard Go-to Point arrives:

  1. Switch the gantry to TRAJ realtime STREAM.
  2. Read payload relative displacement from /payload/pose_e_rel.
  3. Estimate relative velocity.
  4. Command small bounded cart velocities to remove residual swing.

The controller is conservative by default: it limits velocity, limits travel
away from the start point, stops on stale data, and times out.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from scipy.linalg import solve_discrete_are
from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import SetMode
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


class PayloadStabilizer(Node):
    def __init__(self, args):
        super().__init__('payload_stabilizer')
        self.args = args

        self.latest_gantry: GantryState | None = None
        self.latest_payload_time: float | None = None
        self.latest_payload_wall: float | None = None
        self._payload_sequence = 0
        self._awaiting_post_arm_payload = False
        self._arm_payload_sequence = 0
        self._post_arm_deadline_wall: float | None = None
        self.active_axes = ('x', 'y') if args.axis == 'both' else (args.axis,)
        self.latest_swing_mm: dict[str, float | None] = {'x': None, 'y': None}
        self.latest_angle_deg: dict[str, float | None] = {'x': None, 'y': None}
        self.latest_cart_mm: dict[str, float | None] = {'x': None, 'y': None}
        self.latest_cart_vel_mm_s: dict[str, float] = {'x': 0.0, 'y': 0.0}
        self.filtered_swing_mm: dict[str, float | None] = {'x': None, 'y': None}
        self.filtered_vel_mm_s: dict[str, float] = {'x': 0.0, 'y': 0.0}
        self.filter_time: dict[str, float | None] = {'x': None, 'y': None}

        self.start_wall: float | None = None
        self.delay_until_wall: float | None = None
        self.control_active = False
        self.start_cart_mm: dict[str, float | None] = {'x': None, 'y': None}
        self.done = False
        self.aborted = False
        self.arming = False
        self._instant_done_since: float | None = None
        self._last_status_wall = 0.0
        self._rms_window: deque[tuple[float, float]] = deque()
        self._settled_since: float | None = None
        self._last_cmd_mm_s: dict[str, float] = {'x': 0.0, 'y': 0.0}
        self._last_control_wall: float | None = None
        self._last_control_payload_time: float | None = None
        self._last_csv_flush_wall = time.monotonic()
        self._lqg: dict[str, dict[str, np.ndarray]] = {}

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.traj_pub = self.create_publisher(TrajCmd, '/traj_cmd', qos)
        self.mode_cli = self.create_client(SetMode, '/gantry/set_mode')
        self.enable_cli = self.create_client(Trigger, '/gantry/enable')
        self.create_subscription(GantryState, '/gantry/state', self._gantry_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            args.payload_topic,
            self._payload_cb,
            qos_profile_sensor_data,
        )

        self.csv_file = None
        self.csv_writer: csv.writer | None = None
        if args.log_csv:
            self._open_csv_log(args.log_csv)

        self.get_logger().info(
            f'Payload stabilizer ready: controller={args.controller} axis={args.axis} '
            f'max_vel={args.max_vel_mm_s:.1f}mm/s '
            f'max_cart_offset={args.max_cart_offset_mm:.1f}mm timeout={args.timeout_s:.1f}s')

        self.start_timer = self.create_timer(0.2, self._start_when_ready)
        # Control is evaluated directly in the payload callback, once per
        # fresh encoder sample. This timer is only a watchdog; a 100 Hz timer
        # in a single-threaded executor can otherwise starve that callback.
        self.control_timer = self.create_timer(0.05, self._control_tick)

    def destroy_node(self):
        try:
            if rclpy.ok():
                for _ in range(3):
                    self._publish_stream(0.0, 0.0)
                    time.sleep(0.01)
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
            'run_time_sec',
            'axis',
            'swing_x_mm',
            'swing_y_mm',
            'swing_mag_mm',
            'pitch_deg',
            'roll_deg',
            'angle_mag_deg',
            'filtered_x_mm',
            'filtered_y_mm',
            'filtered_vel_x_mm_s',
            'filtered_vel_y_mm_s',
            'cart_x_mm',
            'cart_y_mm',
            'cart_offset_x_mm',
            'cart_offset_y_mm',
            'cart_offset_mag_mm',
            'cmd_vx_mm_s',
            'cmd_vy_mm_s',
            'cmd_mag_mm_s',
            'controller',
            'controller_phase',
            'rms_window_mm',
            'max_abs_window_mm',
            'status',
        ])
        self.csv_file.flush()
        self.get_logger().info(f'CSV log: {path}')

    def _gantry_cb(self, msg: GantryState):
        self.latest_gantry = msg
        self.latest_cart_mm['x'] = 1000.0 * float(msg.x)
        self.latest_cart_mm['y'] = 1000.0 * float(msg.y)
        self.latest_cart_vel_mm_s['x'] = 1000.0 * float(msg.vx)
        self.latest_cart_vel_mm_s['y'] = 1000.0 * float(msg.vy)

    def _payload_cb(self, msg: Float64MultiArray):
        if len(msg.data) <= 4:
            self.get_logger().warn('Payload relative array too short')
            return
        self.latest_payload_time = float(msg.data[0])
        self.latest_payload_wall = time.monotonic()
        self._payload_sequence += 1
        self.latest_swing_mm['x'] = 1000.0 * float(msg.data[3])
        self.latest_swing_mm['y'] = 1000.0 * float(msg.data[4])
        self.latest_angle_deg['x'] = float(msg.data[1])
        self.latest_angle_deg['y'] = float(msg.data[2])
        if (
            self._awaiting_post_arm_payload
            and self._payload_sequence > self._arm_payload_sequence
            and not self.done
        ):
            self._awaiting_post_arm_payload = False
            self._post_arm_deadline_wall = None
            self._start_control()
        if self.control_active and not self.done:
            self._control_tick()

    def _preflight_ready(self) -> bool:
        if self.latest_gantry is None:
            return False
        for axis in self.active_axes:
            if self.latest_cart_mm[axis] is None or self.latest_swing_mm[axis] is None:
                return False
            if self._uses_direct_angle() and self.latest_angle_deg[axis] is None:
                return False
        if self.latest_payload_wall is None:
            return False
        if time.monotonic() - self.latest_payload_wall > self.args.payload_fresh_timeout:
            return False
        if self.latest_gantry.estop:
            return False
        return bool(self.latest_gantry.homed)

    def _request_set_traj_mode(self) -> bool:
        if not self.mode_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/gantry/set_mode not available')
            return False
        req = SetMode.Request()
        req.mode = 'TRAJ'
        future = self.mode_cli.call_async(req)
        future.add_done_callback(self._on_set_traj_done)
        return True

    def _request_enable(self) -> bool:
        if self.latest_gantry is not None and self.latest_gantry.enabled:
            self._wait_for_post_arm_payload()
            return True
        if not self.enable_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/gantry/enable not available')
            return False
        future = self.enable_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_enable_done)
        return True

    def _on_set_traj_done(self, future):
        if self.done:
            return
        try:
            result = future.result()
        except Exception as exc:
            self._abort(f'set TRAJ mode failed: {exc}')
            return
        if result is None or not result.success:
            msg = result.message if result is not None else 'no response'
            self._abort(f'could not enter TRAJ mode: {msg}')
            return
        self.get_logger().info('Gantry set to TRAJ mode for payload stabilization')
        self._publish_stream(0.0, 0.0)
        if not self._request_enable():
            self._abort('could not request /gantry/enable')

    def _on_enable_done(self, future):
        if self.done:
            return
        try:
            result = future.result()
        except Exception as exc:
            self._abort(f'enable failed: {exc}')
            return
        if result is None or not result.success:
            msg = result.message if result is not None else 'no response'
            self._abort(f'could not enable TRAJ realtime stream: {msg}')
            return
        self.get_logger().info(f'/gantry/enable: {result.message}')
        self._wait_for_post_arm_payload()

    def _wait_for_post_arm_payload(self):
        """Begin only from a sample received after TRAJ is completely armed."""
        if self.done or self.start_wall is not None:
            return
        self._awaiting_post_arm_payload = True
        self._arm_payload_sequence = self._payload_sequence
        self._post_arm_deadline_wall = time.monotonic() + self.args.post_arm_payload_timeout_s
        self.get_logger().info(
            'TRAJ armed; waiting for a new payload encoder sample before stabilization')

    def _start_control(self):
        if self.done or self.start_wall is not None:
            return
        # Entering TRAJ and enabling motors can take longer than the payload
        # freshness budget. Wait for the next encoder sample instead of
        # declaring a false stale-data fault immediately after arming.
        if not self._preflight_ready():
            self.arming = False
            self.get_logger().info('Armed; waiting for a fresh payload sample before stabilization')
            return
        try:
            self.start_timer.cancel()
        except Exception:
            pass
        if self.args.start_delay_s > 0.0:
            self.start_wall = time.monotonic()
            self.delay_until_wall = self.start_wall + self.args.start_delay_s
            self.get_logger().info(
                f'Observing payload for {self.args.start_delay_s:.2f}s before damping')
            return
        self._activate_control()

    def _activate_control(self):
        if self.done or self.control_active:
            return
        for axis in self.active_axes:
            self.start_cart_mm[axis] = self.latest_cart_mm[axis]
            self.filtered_swing_mm[axis] = self.latest_swing_mm[axis]
            self.filtered_vel_mm_s[axis] = 0.0
            self.filter_time[axis] = self.latest_payload_time
        # Build the controller before going live. The prior iterative Riccati
        # solve could take longer than the payload freshness watchdog.
        if self.args.controller == 'lqg':
            self._initialize_lqg()
        self.control_active = True
        self.start_wall = time.monotonic()
        # _preflight_ready() has just accepted this measurement. Latch its
        # freshness at activation so a control-timer callback in the same ROS
        # spin cycle cannot reject the same sample before the next 100 Hz
        # encoder message arrives. The normal watchdog still aborts after the
        # configured timeout if that next message never arrives.
        self.latest_payload_wall = self.start_wall
        self._last_cmd_mm_s = {'x': 0.0, 'y': 0.0}
        self._last_control_wall = time.monotonic()
        self._last_control_payload_time = None
        self._settled_since = None
        self.get_logger().info(
            'Stabilization started: '
            f'cart=({self._value_text(self.start_cart_mm["x"])},'
            f'{self._value_text(self.start_cart_mm["y"])})mm '
            f'angle=({self._value_text(self.latest_angle_deg["x"])},'
            f'{self._value_text(self.latest_angle_deg["y"])})deg')

    def _start_when_ready(self):
        if self.done or self.arming or self.start_wall is not None:
            return
        if not self._preflight_ready():
            now = time.monotonic()
            if now - self._last_status_wall > 1.0:
                self._last_status_wall = now
                self.get_logger().info('Waiting for homed gantry state and fresh payload data')
            return
        if not self.args.no_arm:
            self.arming = True
            if not self._request_set_traj_mode():
                self._abort('could not enter TRAJ mode')
                return
            return
        self._start_control()

    def _control_tick(self):
        if self.done:
            return
        if self.start_wall is None:
            if (
                self._awaiting_post_arm_payload
                and self._post_arm_deadline_wall is not None
                and time.monotonic() >= self._post_arm_deadline_wall
            ):
                self._abort(
                    f'no new payload sample within '
                    f'{self.args.post_arm_payload_timeout_s:.2f}s after TRAJ arming')
            return

        now = time.monotonic()
        if not self.control_active:
            self._publish_stream(0.0, 0.0)
            if self.delay_until_wall is not None and now >= self.delay_until_wall:
                self._activate_control()
            return

        run_t = now - self.start_wall
        if run_t >= self.args.timeout_s:
            self._finish('timeout')
            return
        if self.latest_payload_wall is None or now - self.latest_payload_wall > self.args.payload_fresh_timeout:
            age_ms = float('nan') if self.latest_payload_wall is None else (
                now - self.latest_payload_wall) * 1000.0
            self._abort(
                f'payload data stale (age={age_ms:.1f}ms '
                f'limit={self.args.payload_fresh_timeout * 1000.0:.1f}ms)')
            return
        if self.latest_gantry is None or self.latest_gantry.estop:
            self._abort('gantry state unavailable or E-stop active')
            return

        # The watchdog may run between encoder samples. Do not differentiate,
        # update the observer, or resend a new command from a repeated sample.
        if self.latest_payload_time is None:
            return
        if self.latest_payload_time == self._last_control_payload_time:
            return
        self._last_control_payload_time = self.latest_payload_time

        swing = self._active_values(self.latest_swing_mm)
        angle = self._active_values(self.latest_angle_deg)
        filt: dict[str, float] = {}
        vel: dict[str, float] = {}
        for axis in self.active_axes:
            source = angle[axis] if self._uses_direct_angle() else swing[axis]
            filt[axis], vel[axis] = self._filter_swing(axis, source)
        cart_offset = self._cart_offset_mm()
        swing_mag = self._vector_mag(swing)
        angle_mag = self._vector_mag(angle)
        cart_offset_mag = self._vector_mag(cart_offset)
        metric = angle_mag if self._uses_direct_angle() else swing_mag
        metric_name = 'deg' if self._uses_direct_angle() else 'mm'
        done_rms_limit = self.args.done_rms_deg if self._uses_direct_angle() else self.args.done_rms_mm
        done_max_limit = self.args.done_max_abs_deg if self._uses_direct_angle() else self.args.done_max_abs_mm

        if self.args.controller == 'legacy-pd':
            cmd, status = self._legacy_pd_command(filt, vel, cart_offset, cart_offset_mag)
        elif self.args.controller == 'lqg':
            cmd, status = self._lqg_command(filt, vel, cart_offset, cart_offset_mag)
        else:
            cmd, status = self._phased_pd_command(filt, vel, cart_offset, cart_offset_mag)
        cmd = self._slew_limit_command(cmd, now)
        if self.args.controller == 'lqg':
            # The observer must use the constrained command actually sent to
            # the motor, not its pre-saturation request.
            for axis in self.active_axes:
                self._lqg[axis]['u'][0, 0] = cmd[axis] / 1000.0

        vx = cmd.get('x', 0.0)
        vy = cmd.get('y', 0.0)
        rms, max_abs = self._update_window(run_t, metric)

        if now - self._last_status_wall >= self.args.print_period:
            self._last_status_wall = now
            self.get_logger().info(
                f't={run_t:.2f}s angle=({angle.get("x", 0.0):+.3f},'
                f'{angle.get("y", 0.0):+.3f})deg |mag|={angle_mag:.3f}deg '
                f'angle_rate=({vel.get("x", 0.0):+.2f},{vel.get("y", 0.0):+.2f})deg/s '
                f'cart_offset=({cart_offset.get("x", 0.0):+.2f},'
                f'{cart_offset.get("y", 0.0):+.2f})mm '
                f'cmd=({vx:+.1f},{vy:+.1f})mm/s rms={rms:.3f}{metric_name} status={status}')

        if (
            run_t >= self.args.min_run_s
            and cart_offset_mag <= self.args.final_cart_tolerance_mm
            and self._instant_done_ready(run_t, metric)
        ):
            self._log_row(
                run_t, swing, angle, filt, vel, cart_offset,
                {'x': 0.0, 'y': 0.0}, rms, max_abs, 'instant_settled',
            )
            self._finish('instant settled')
            return

        self._publish_stream(vx, vy)
        self._log_row(run_t, swing, angle, filt, vel, cart_offset, cmd, rms, max_abs, status)

        if (
            run_t >= self.args.min_run_s
            and (not self.args.return_cart_after_settle or cart_offset_mag <= self.args.final_cart_tolerance_mm)
            and len(self._rms_window) >= max(3, int(0.5 * self.args.rate_hz))
            and rms <= done_rms_limit
            and max_abs <= done_max_limit
        ):
            self._finish('settled')

    def _filter_swing(self, axis: str, swing_mm: float) -> tuple[float, float]:
        t = self.latest_payload_time
        if t is None:
            return swing_mm, self.filtered_vel_mm_s[axis]
        if self.filtered_swing_mm[axis] is None or self.filter_time[axis] is None:
            self.filtered_swing_mm[axis] = swing_mm
            self.filter_time[axis] = t
            self.filtered_vel_mm_s[axis] = 0.0
            return self.filtered_swing_mm[axis], self.filtered_vel_mm_s[axis]

        dt = max(0.0, t - self.filter_time[axis])
        if dt <= 1.0e-6:
            return self.filtered_swing_mm[axis], self.filtered_vel_mm_s[axis]
        alpha = 1.0
        if self.args.lowpass_hz > 0.0:
            alpha = 1.0 - math.exp(-2.0 * math.pi * self.args.lowpass_hz * dt)
            alpha = max(0.0, min(alpha, 1.0))
        prev = self.filtered_swing_mm[axis]
        self.filtered_swing_mm[axis] = prev + alpha * (swing_mm - prev)
        raw_vel = (self.filtered_swing_mm[axis] - prev) / dt
        v_alpha = 1.0
        if self.args.velocity_lowpass_hz > 0.0:
            v_alpha = 1.0 - math.exp(-2.0 * math.pi * self.args.velocity_lowpass_hz * dt)
            v_alpha = max(0.0, min(v_alpha, 1.0))
        self.filtered_vel_mm_s[axis] += v_alpha * (raw_vel - self.filtered_vel_mm_s[axis])
        self.filter_time[axis] = t
        return self.filtered_swing_mm[axis], self.filtered_vel_mm_s[axis]

    def _uses_direct_angle(self) -> bool:
        return self.args.controller in ('phased-pd', 'lqg')

    def _legacy_pd_command(
        self,
        filt: dict[str, float],
        vel: dict[str, float],
        cart_offset: dict[str, float],
        cart_offset_mag: float,
    ) -> tuple[dict[str, float], str]:
        """Preserve the previous displacement-PD behavior for A/B comparison."""
        if cart_offset_mag >= self.args.max_cart_offset_mm:
            return self._scale_vector(
                {axis: -cart_offset[axis] for axis in self.active_axes},
                self.args.max_vel_mm_s,
            ), 'cart_offset_limit'
        raw = {
            axis: self._control_sign(axis) * (
                self.args.swing_gain_s_inv * filt[axis]
                + self.args.swing_velocity_gain * vel[axis]
            ) - self.args.cart_return_gain_s_inv * cart_offset[axis]
            for axis in self.active_axes
        }
        return self._limit_vector(raw, self.args.max_vel_mm_s), 'legacy_pd'

    def _phased_pd_command(
        self,
        angle_deg: dict[str, float],
        rate_deg_s: dict[str, float],
        cart_offset: dict[str, float],
        cart_offset_mag: float,
    ) -> tuple[dict[str, float], str]:
        """Direct-angle PD; cart return is deferred until the payload is quiet."""
        quiet = (
            self._vector_mag(angle_deg) <= self.args.angle_deadband_deg
            and self._vector_mag(rate_deg_s) <= self.args.angle_rate_deadband_deg_s
        )
        if quiet:
            if self._settled_since is None:
                self._settled_since = time.monotonic()
        else:
            self._settled_since = None

        if (
            self.args.return_cart_after_settle
            and self._settled_since is not None
            and time.monotonic() - self._settled_since >= self.args.settle_hold_s
        ):
            if cart_offset_mag <= self.args.final_cart_tolerance_mm:
                return {'x': 0.0, 'y': 0.0}, 'cart_return_complete'
            return self._limit_vector(
                {axis: -self.args.cart_return_gain_s_inv * cart_offset[axis]
                 for axis in self.active_axes},
                self.args.max_return_vel_mm_s,
            ), 'cart_return_after_settle'

        if cart_offset_mag >= self.args.max_cart_offset_mm:
            # Stop increasing excursion, but do not reverse aggressively into a swing.
            return {'x': 0.0, 'y': 0.0}, 'cart_offset_hold'
        raw = {
            axis: self._control_sign(axis) * (
                self.args.angle_gain_mm_s_per_deg * angle_deg[axis]
                + self.args.angle_rate_gain_mm_s_per_deg_s * rate_deg_s[axis]
            )
            for axis in self.active_axes
        }
        return self._limit_vector(raw, self.args.max_vel_mm_s), 'phased_pd'

    def _initialize_lqg(self):
        """Build one discrete LQG controller per active Cartesian axis.

        The cart accepts a velocity command; its first-order velocity response is
        included explicitly, so the pendulum is driven by modeled cart
        acceleration rather than by an unrealistic instantaneous velocity step.
        """
        dt = 1.0 / self.args.rate_hz
        omega = math.sqrt(9.80665 / self.args.rope_length_m)
        tau = self.args.actuator_time_constant_s
        zeta = self.args.model_zeta
        for axis in self.active_axes:
            sign = self._control_sign(axis)
            ac = np.array([
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0 / tau, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, sign / (self.args.rope_length_m * tau), -omega * omega,
                 -2.0 * zeta * omega],
            ])
            bc = np.array([[0.0], [sign / tau], [0.0],
                           [-sign / (self.args.rope_length_m * tau)]])
            ad = np.eye(4) + dt * ac
            bd = dt * bc
            q = np.diag([
                self.args.lqg_q_cart_position,
                self.args.lqg_q_cart_velocity,
                self.args.lqg_q_angle,
                self.args.lqg_q_angle_rate,
            ])
            k = self._discrete_lqr(ad, bd, q, np.array([[self.args.lqg_r_command]]))
            c = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ])
            theta_std = math.radians(self.args.angle_measurement_std_deg)
            self._lqg[axis] = {
                'A': ad, 'B': bd, 'K': k, 'C': c,
                'Qn': np.diag([1.0e-9, 3.0e-5, 1.0e-8, 4.0e-4]),
                'Rn': np.diag([2.5e-9, 4.0e-5, theta_std * theta_std]),
                'x': np.array([[0.0], [self.latest_cart_vel_mm_s[axis] / 1000.0],
                               [math.radians(self.latest_angle_deg[axis] or 0.0)], [0.0]]),
                'P': np.diag([1.0e-5, 2.0e-3, 2.0e-5, 2.0e-3]),
                'u': np.array([[0.0]]),
            }
        self.get_logger().info(
            f'LQG initialized: L={self.args.rope_length_m:.3f}m '
            f'omega={omega:.3f}rad/s actuator_tau={tau * 1000.0:.0f}ms')

    @staticmethod
    def _discrete_lqr(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        p = solve_discrete_are(a, b, q, r)
        return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)

    def _lqg_command(
        self,
        angle_deg: dict[str, float],
        _rate_deg_s: dict[str, float],
        cart_offset: dict[str, float],
        cart_offset_mag: float,
    ) -> tuple[dict[str, float], str]:
        if cart_offset_mag >= self.args.max_cart_offset_mm:
            return {'x': 0.0, 'y': 0.0}, 'cart_offset_hold'
        cmd = {'x': 0.0, 'y': 0.0}
        for axis in self.active_axes:
            f = self._lqg[axis]
            x_pred = f['A'] @ f['x'] + f['B'] @ f['u']
            p_pred = f['A'] @ f['P'] @ f['A'].T + f['Qn']
            measurement = np.array([
                [cart_offset[axis] / 1000.0],
                [self.latest_cart_vel_mm_s[axis] / 1000.0],
                [math.radians(angle_deg[axis])],
            ])
            innovation_cov = f['C'] @ p_pred @ f['C'].T + f['Rn']
            gain = np.linalg.solve(innovation_cov, f['C'] @ p_pred).T
            f['x'] = x_pred + gain @ (measurement - f['C'] @ x_pred)
            f['P'] = (np.eye(4) - gain @ f['C']) @ p_pred
            f['u'] = -f['K'] @ f['x']
            cmd[axis] = float(f['u'][0, 0]) * 1000.0
        return self._limit_vector(cmd, self.args.max_vel_mm_s), 'lqg'

    def _slew_limit_command(self, cmd: dict[str, float], now: float) -> dict[str, float]:
        if self._last_control_wall is None:
            dt = 1.0 / self.args.rate_hz
        else:
            dt = min(max(now - self._last_control_wall, 1.0e-3), 0.1)
        delta_limit = self.args.max_accel_mm_s2 * dt
        out = {'x': 0.0, 'y': 0.0}
        for axis in self.active_axes:
            target = float(cmd.get(axis, 0.0))
            delta = max(-delta_limit, min(delta_limit, target - self._last_cmd_mm_s[axis]))
            out[axis] = self._last_cmd_mm_s[axis] + delta
            self._last_cmd_mm_s[axis] = out[axis]
        self._last_control_wall = now
        return out

    def _cart_offset_mm(self) -> dict[str, float]:
        out = {'x': 0.0, 'y': 0.0}
        for axis in self.active_axes:
            if self.start_cart_mm[axis] is None or self.latest_cart_mm[axis] is None:
                out[axis] = 0.0
            else:
                out[axis] = self.latest_cart_mm[axis] - self.start_cart_mm[axis]
        return out

    def _active_values(self, values: dict[str, float | None]) -> dict[str, float]:
        return {axis: 0.0 if values[axis] is None else float(values[axis]) for axis in self.active_axes}

    def _vector_mag(self, values: dict[str, float]) -> float:
        return math.sqrt(sum(values.get(axis, 0.0) ** 2 for axis in self.active_axes))

    def _limit_vector(self, values: dict[str, float], limit: float) -> dict[str, float]:
        out = {'x': 0.0, 'y': 0.0}
        for axis in self.active_axes:
            out[axis] = float(values.get(axis, 0.0))
        mag = self._vector_mag(out)
        if mag > limit and mag > 1.0e-9:
            scale = limit / mag
            for axis in self.active_axes:
                out[axis] *= scale
        return out

    def _scale_vector(self, values: dict[str, float], target_mag: float) -> dict[str, float]:
        out = {'x': 0.0, 'y': 0.0}
        mag = math.sqrt(sum(values.get(axis, 0.0) ** 2 for axis in self.active_axes))
        if mag <= 1.0e-9:
            return out
        scale = target_mag / mag
        for axis in self.active_axes:
            out[axis] = values.get(axis, 0.0) * scale
        return out

    def _value_text(self, value: float | None) -> str:
        return 'nan' if value is None else f'{value:.2f}'

    def _control_sign(self, axis: str) -> float:
        if axis == 'x':
            return self.args.control_sign_x
        if axis == 'y':
            return self.args.control_sign_y
        return self.args.control_sign

    def _update_window(self, run_t: float, swing_mag_mm: float) -> tuple[float, float]:
        self._rms_window.append((run_t, swing_mag_mm))
        cutoff = run_t - self.args.done_window_s
        while self._rms_window and self._rms_window[0][0] < cutoff:
            self._rms_window.popleft()
        values = [x for _, x in self._rms_window]
        if not values:
            return float('nan'), float('nan')
        rms = math.sqrt(sum(x * x for x in values) / len(values))
        max_abs = max(abs(x) for x in values)
        return rms, max_abs

    def _instant_done_ready(self, run_t: float, swing_mag_mm: float) -> bool:
        if self.args.instant_done_max_abs_mm <= 0.0:
            return False
        if swing_mag_mm > self.args.instant_done_max_abs_mm:
            self._instant_done_since = None
            return False
        if self._instant_done_since is None:
            self._instant_done_since = run_t
            return self.args.instant_done_hold_s <= 0.0
        return run_t - self._instant_done_since >= self.args.instant_done_hold_s

    def _publish_stream(self, vx_mm_s: float, vy_mm_s: float):
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = TrajCmd.STREAM
        msg.vx_mm_s = float(vx_mm_s)
        msg.vy_mm_s = float(vy_mm_s)
        if self.latest_gantry is not None:
            msg.x = float(self.latest_gantry.x)
            msg.y = float(self.latest_gantry.y)
        self.traj_pub.publish(msg)

    def _log_row(
        self,
        run_t: float,
        swing: dict[str, float],
        angle: dict[str, float],
        filt: dict[str, float],
        vel: dict[str, float],
        cart_offset: dict[str, float],
        cmd: dict[str, float],
        rms: float,
        max_abs: float,
        status: str,
    ):
        if self.csv_writer is None:
            return
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds * 1.0e-9,
            run_t,
            self.args.axis,
            swing.get('x', 0.0),
            swing.get('y', 0.0),
            self._vector_mag(swing),
            angle.get('x', 0.0),
            angle.get('y', 0.0),
            self._vector_mag(angle),
            filt.get('x', 0.0),
            filt.get('y', 0.0),
            vel.get('x', 0.0),
            vel.get('y', 0.0),
            '' if self.latest_cart_mm['x'] is None else self.latest_cart_mm['x'],
            '' if self.latest_cart_mm['y'] is None else self.latest_cart_mm['y'],
            cart_offset.get('x', 0.0),
            cart_offset.get('y', 0.0),
            self._vector_mag(cart_offset),
            cmd.get('x', 0.0),
            cmd.get('y', 0.0),
            self._vector_mag(cmd),
            self.args.controller,
            'direct_angle' if self._uses_direct_angle() else 'legacy_displacement',
            rms,
            max_abs,
            status,
        ])
        # The control path runs on each encoder sample.  Flushing every row
        # can occasionally block long enough to delay DDS callbacks, which
        # then looks like a false payload-stale fault.  Keep writes buffered
        # and commit them at a modest cadence; destroy_node() closes the file.
        if (
            self.csv_file is not None
            and time.monotonic() - self._last_csv_flush_wall >= 0.5
        ):
            self.csv_file.flush()
            self._last_csv_flush_wall = time.monotonic()

    def _abort(self, reason: str):
        if self.done:
            return
        self.aborted = True
        self._publish_stream(0.0, 0.0)
        self.get_logger().error(f'Stabilization aborted: {reason}')
        if self.csv_file is not None:
            self.csv_file.flush()
        self.done = True

    def _finish(self, reason: str):
        if self.done:
            return
        self._publish_stream(0.0, 0.0)
        cart_offset = self._cart_offset_mm()
        values = [value for _, value in self._rms_window]
        if values:
            rms = math.sqrt(sum(value * value for value in values) / len(values))
            max_abs = max(abs(value) for value in values)
        else:
            rms = float('nan')
            max_abs = float('nan')
        units = 'deg' if self._uses_direct_angle() else 'mm'
        self.get_logger().info(
            f'Payload stabilization complete: {reason} | '
            f'cart_offset=({cart_offset.get("x", 0.0):+.2f},'
            f'{cart_offset.get("y", 0.0):+.2f})mm '
            f'rms={rms:.3f}{units} max_abs={max_abs:.3f}{units}')
        if self.csv_file is not None:
            self.csv_file.flush()
        self.done = True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis', choices=('x', 'y', 'both'), default='both')
    p.add_argument(
        '--controller',
        choices=('phased-pd', 'lqg', 'legacy-pd'),
        default='phased-pd',
        help=('phased-pd uses direct encoder angles and delays cart return; '
              'lqg enables the constrained model-based controller; '
              'legacy-pd preserves the previous displacement controller.'),
    )
    p.add_argument('--payload-topic', default='/payload/pose_e_rel')
    p.add_argument('--rate-hz', type=float, default=100.0)
    p.add_argument('--timeout-s', type=float, default=10.0)
    p.add_argument(
        '--start-delay-s',
        type=float,
        default=0.0,
        help='Observe with zero STREAM for this long before starting active damping.',
    )
    p.add_argument('--min-run-s', type=float, default=1.0)
    p.add_argument('--payload-fresh-timeout', type=float, default=0.25)
    p.add_argument(
        '--post-arm-payload-timeout-s', type=float, default=2.0,
        help='Maximum wait for a new payload sample after TRAJ is armed; no motion occurs while waiting.',
    )
    p.add_argument('--print-period', type=float, default=0.25)
    p.add_argument('--max-vel-mm-s', type=float, default=45.0)
    p.add_argument(
        '--max-accel-mm-s2', type=float, default=180.0,
        help='Maximum command-velocity slew rate. Prevents reversal-induced re-excitation.',
    )
    p.add_argument('--max-return-vel-mm-s', type=float, default=20.0)
    p.add_argument('--max-cart-offset-mm', type=float, default=35.0)
    p.add_argument('--final-cart-tolerance-mm', type=float, default=8.0)
    p.add_argument('--swing-gain-s-inv', type=float, default=1.0)
    p.add_argument('--swing-velocity-gain', type=float, default=0.06)
    p.add_argument(
        '--angle-gain-mm-s-per-deg', type=float, default=10.0,
        help='Direct-angle proportional gain used by phased-pd.',
    )
    p.add_argument(
        '--angle-rate-gain-mm-s-per-deg-s', type=float, default=1.5,
        help='Direct-angle-rate gain used by phased-pd.',
    )
    p.add_argument('--cart-return-gain-s-inv', type=float, default=0.7)
    p.add_argument(
        '--control-sign',
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help='Default control sign for axes without an axis-specific override.',
    )
    p.add_argument(
        '--control-sign-x',
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help='X-axis control sign. Flip if X damping injects energy.',
    )
    p.add_argument(
        '--control-sign-y',
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help='Y-axis control sign. Flip if Y damping injects energy.',
    )
    p.add_argument('--deadband-mm', type=float, default=1.0)
    p.add_argument('--velocity-deadband-mm-s', type=float, default=3.0)
    p.add_argument('--angle-deadband-deg', type=float, default=0.18)
    p.add_argument('--angle-rate-deadband-deg-s', type=float, default=0.50)
    p.add_argument('--lowpass-hz', type=float, default=5.0)
    p.add_argument('--velocity-lowpass-hz', type=float, default=4.0)
    p.add_argument('--done-window-s', type=float, default=1.0)
    p.add_argument('--done-rms-mm', type=float, default=5.0)
    p.add_argument('--done-max-abs-mm', type=float, default=10.0)
    p.add_argument('--done-rms-deg', type=float, default=0.18)
    p.add_argument('--done-max-abs-deg', type=float, default=0.36)
    p.add_argument('--settle-hold-s', type=float, default=0.50)
    p.add_argument(
        '--return-cart-after-settle', action='store_true',
        help='Return to the initial cart position only after angle and rate are quiet.',
    )
    p.add_argument('--rope-length-m', type=float, default=0.90)
    p.add_argument('--model-zeta', type=float, default=0.02)
    p.add_argument('--actuator-time-constant-s', type=float, default=0.045)
    p.add_argument('--angle-measurement-std-deg', type=float, default=0.06)
    p.add_argument('--lqg-q-cart-position', type=float, default=5.0)
    p.add_argument('--lqg-q-cart-velocity', type=float, default=0.5)
    p.add_argument('--lqg-q-angle', type=float, default=300.0)
    p.add_argument('--lqg-q-angle-rate', type=float, default=20.0)
    p.add_argument('--lqg-r-command', type=float, default=8.0)
    p.add_argument(
        '--instant-done-max-abs-mm',
        type=float,
        default=0.0,
        help='If >0, stop once instantaneous vector swing magnitude stays below this threshold.',
    )
    p.add_argument('--instant-done-hold-s', type=float, default=0.08)
    p.add_argument('--log-csv', default='')
    p.add_argument('--no-arm', action='store_true', help='Do not set TRAJ mode or call enable.')
    return p.parse_args()


def check_args(args) -> bool:
    checks = [
        ('--rate-hz', args.rate_hz > 0.0),
        ('--timeout-s', args.timeout_s > 0.0),
        ('--start-delay-s', args.start_delay_s >= 0.0),
        ('--min-run-s', args.min_run_s >= 0.0),
        ('--payload-fresh-timeout', args.payload_fresh_timeout > 0.0),
        ('--post-arm-payload-timeout-s', args.post_arm_payload_timeout_s > 0.0),
        ('--print-period', args.print_period > 0.0),
        ('--max-vel-mm-s', args.max_vel_mm_s > 0.0),
        ('--max-accel-mm-s2', args.max_accel_mm_s2 > 0.0),
        ('--max-return-vel-mm-s', args.max_return_vel_mm_s > 0.0),
        ('--max-cart-offset-mm', args.max_cart_offset_mm > 0.0),
        ('--final-cart-tolerance-mm', args.final_cart_tolerance_mm >= 0.0),
        ('--done-window-s', args.done_window_s > 0.0),
        ('--done-rms-mm', args.done_rms_mm > 0.0),
        ('--done-max-abs-mm', args.done_max_abs_mm > 0.0),
        ('--angle-deadband-deg', args.angle_deadband_deg > 0.0),
        ('--angle-rate-deadband-deg-s', args.angle_rate_deadband_deg_s > 0.0),
        ('--done-rms-deg', args.done_rms_deg > 0.0),
        ('--done-max-abs-deg', args.done_max_abs_deg > 0.0),
        ('--settle-hold-s', args.settle_hold_s >= 0.0),
        ('--rope-length-m', args.rope_length_m > 0.0),
        ('--model-zeta', 0.0 <= args.model_zeta < 1.0),
        ('--actuator-time-constant-s', args.actuator_time_constant_s > 0.0),
        ('--angle-measurement-std-deg', args.angle_measurement_std_deg > 0.0),
        ('--lqg-q-cart-position', args.lqg_q_cart_position >= 0.0),
        ('--lqg-q-cart-velocity', args.lqg_q_cart_velocity >= 0.0),
        ('--lqg-q-angle', args.lqg_q_angle > 0.0),
        ('--lqg-q-angle-rate', args.lqg_q_angle >= 0.0),
        ('--lqg-r-command', args.lqg_r_command > 0.0),
        ('--instant-done-max-abs-mm', args.instant_done_max_abs_mm >= 0.0),
        ('--instant-done-hold-s', args.instant_done_hold_s >= 0.0),
    ]
    for name, ok in checks:
        if not ok:
            print(f'Refusing: {name} is invalid', file=sys.stderr)
            return False
    return True


def main():
    args = parse_args()
    if not check_args(args):
        return 1
    rclpy.init()
    node = None
    try:
        node = PayloadStabilizer(args)
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[payload_stabilizer] {exc}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
