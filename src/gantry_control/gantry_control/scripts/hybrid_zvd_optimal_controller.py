#!/usr/bin/env python3
"""
Joystick pulse or ZVD-front/minimum-time-open-loop-stop hybrid jog.

Button 10 toggles PULSE and HYBRID while idle.  PULSE is a conventional direct
velocity jog.  In HYBRID, hold LB and move the left stick to begin a
ZVD-shaped jog.  While the stick remains held, the measured cart/sway observer
is continuously forecast and a robust "stop if released now" linear program
is refreshed in the background.  Releasing the stick immediately plays the
freshest safe plan.  If no current plan is available, a deterministic ZV
release tail keeps half speed for one switch interval (for zero damping)
before commanding zero.

The stop has no fixed endpoint: the cart is allowed to move a short distance
forward while it removes its velocity and payload sway.  The LP searches for
the shortest feasible horizon and hard-constrains every velocity command to
the incoming direction or zero, so repeated cart reversals are impossible.
Measurements during playback are safety/model-validation signals only and do
not turn the stop into a feedback controller.

The filename is retained for compatibility with existing lab commands.

RB requests the gantry E-stop.  This node owns /traj_cmd while running, so do
not run it alongside another TRAJ streaming player.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import SetMode
from optimal_stop_planner import (
    CraneModel,
    CraneStateObserver,
    ForwardOnlyMinimumTimeStopPlanner,
    MinimumTimeStopConfig,
    MinimumTimeStopPlan,
    OpenLoopPlanningError,
    discrete_crane_model,
)


G = 9.80665


class HybridZvdOptimalJog(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('hybrid_zvd_optimal_jog')
        self.args = args
        self.phase = 'arming'
        self.selected_mode = args.start_mode
        self.last_mode_button = False
        self.latest_gantry: GantryState | None = None
        self.gantry_wall: float | None = None
        self.gantry_history: deque[tuple[float, float, float]] = deque(maxlen=200)
        self.cart_mm = {'x': None, 'y': None}
        self.cart_vel_mm_s = {'x': 0.0, 'y': 0.0}
        self.angle_deg = {'x': None, 'y': None}
        self.payload_wall: float | None = None
        self.payload_time: float | None = None
        self.last_payload_time: float | None = None
        self.joy_dir = {'x': 0.0, 'y': 0.0}
        self.joy_magnitude = 0.0
        self.stick_active = False
        self.last_joy_wall: float | None = None
        self.transition_wall: float | None = None
        self.target_cart_mm = {'x': None, 'y': None}
        model = CraneModel(
            rope_length_m=args.rope_length_m,
            damping_ratio=args.model_zeta,
            actuator_time_constant_s=args.actuator_tau_s,
        )
        self._forecast_model = model
        self._planning_forecast_s = args.optimal_plan_forecast_s
        self.observers = {
            axis: CraneStateObserver(
                model,
                angle_measurement_std_rad=math.radians(args.angle_measurement_std_deg),
            )
            for axis in ('x', 'y')
        }
        self.planner = ForwardOnlyMinimumTimeStopPlanner(MinimumTimeStopConfig(
            sample_period_s=args.optimal_sample_period_s,
            minimum_horizon_s=args.optimal_min_horizon_s,
            maximum_horizon_s=args.optimal_horizon_s,
            horizon_resolution_s=args.optimal_horizon_resolution_s,
            rope_length_m=args.rope_length_m,
            damping_ratio=args.model_zeta,
            actuator_time_constant_s=args.actuator_tau_s,
            rope_length_uncertainty_fraction=args.rope_length_uncertainty_fraction,
            actuator_uncertainty_fraction=args.actuator_uncertainty_fraction,
            uncertainty_samples=args.uncertainty_samples,
            max_command_speed_m_s=args.jog_speed_mm_s / 1000.0,
            max_command_acceleration_m_s2=args.max_regulate_accel_mm_s2 / 1000.0,
            terminal_velocity_tolerance_m_s=args.quiescent_cart_vel_mm_s / 1000.0,
            terminal_angle_tolerance_rad=math.radians(args.quiescent_angle_deg),
            robust_terminal_tolerance_multiplier=args.robust_terminal_multiplier,
            solver_time_limit_s=args.optimizer_time_limit_s,
        ))
        self._planner_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='open_loop_stop_planner'
        )
        self._plan_future: Future[MinimumTimeStopPlan] | None = None
        self._plan_future_meta: dict | None = None
        self._cached_stop_plan: MinimumTimeStopPlan | None = None
        self._cached_plan_meta: dict | None = None
        self._last_plan_submit_wall = -math.inf
        self.open_loop_plan: MinimumTimeStopPlan | None = None
        self.plan_start_wall: float | None = None
        self.plan_origin_cart_mm = {'x': None, 'y': None}
        self.last_applied_cmd = {'x': 0.0, 'y': 0.0}
        self.zv_stop_cmd = {'x': 0.0, 'y': 0.0}
        self.zv_stop_reason = ''
        self.rms_window: deque[tuple[float, float]] = deque()
        self.verification_start: float | None = None
        self._quiescent_since: float | None = None
        self.last_status_wall = 0.0
        self._last_plan_report_wall = 0.0
        self.done = False
        self._last_csv_flush_wall = time.monotonic()

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.traj_pub = self.create_publisher(TrajCmd, '/traj_cmd', qos)
        self.create_subscription(GantryState, '/gantry/state', self._gantry_cb, 10)
        self.create_subscription(Float64MultiArray, args.payload_topic, self._payload_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Joy, '/joy', self._joy_cb, 10)
        self.mode_cli = self.create_client(SetMode, '/gantry/set_mode')
        self.enable_cli = self.create_client(Trigger, '/gantry/enable')
        self.estop_cli = self.create_client(Trigger, '/gantry/estop')

        self.csv_file = None
        self.csv_writer = None
        if args.log_csv:
            path = Path(args.log_csv).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = path.open('w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'wall_time_sec', 'phase', 'target_x_mm', 'target_y_mm',
                'cart_x_mm', 'cart_y_mm', 'cart_error_x_mm', 'cart_error_y_mm',
                'pitch_deg', 'roll_deg', 'angle_mag_deg', 'cmd_vx_mm_s',
                'cmd_vy_mm_s', 'rms_window_deg', 'max_abs_window_deg',
            ])

        self.arm_timer = self.create_timer(0.2, self._arm_when_ready)
        self.timer = self.create_timer(1.0 / args.rate_hz, self._timer_cb)
        self.get_logger().info(
            f'Pulse/Hybrid jog ready: start={self.selected_mode.upper()} '
            f'L={args.rope_length_m:.3f}m '
            f'ZVD T={self.zvd_T:.3f}s speed={args.jog_speed_mm_s:.1f}mm/s; '
            f'forward-only minimum-time stop={args.optimal_min_horizon_s:.2f}-'
            f'{args.optimal_horizon_s:.2f}s/{len(self.planner.scenarios)} models, '
            f'replan={args.optimal_replan_period_s:.2f}s. '
            'Button 10 toggles Pulse/Hybrid while idle. Hold LB + left stick to move.')

    @property
    def active_axes(self) -> tuple[str, ...]:
        return ('x', 'y') if self.args.axis == 'both' else (self.args.axis,)

    @property
    def zvd_T(self) -> float:
        if self.args.zvd_t_s > 0.0:
            return self.args.zvd_t_s
        return math.pi / math.sqrt(G / self.args.rope_length_m)

    def _gantry_cb(self, msg: GantryState):
        self.latest_gantry = msg
        self.gantry_wall = time.monotonic()
        self.cart_mm['x'] = 1000.0 * float(msg.x)
        self.cart_mm['y'] = 1000.0 * float(msg.y)
        self.cart_vel_mm_s['x'] = 1000.0 * float(msg.vx)
        self.cart_vel_mm_s['y'] = 1000.0 * float(msg.vy)
        stamp = float(msg.header.stamp.sec) + 1.0e-9 * float(msg.header.stamp.nanosec)
        if stamp > 0.0:
            self.gantry_history.append((stamp, self.cart_mm['x'], self.cart_mm['y']))

    def _payload_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 3:
            return
        self.payload_time = float(msg.data[0])
        self.payload_wall = time.monotonic()
        self.angle_deg['x'] = float(msg.data[1])
        self.angle_deg['y'] = float(msg.data[2])
        if self.payload_time != self.last_payload_time:
            self.last_payload_time = self.payload_time
            if all(self.cart_mm[axis] is not None for axis in ('x', 'y')):
                for axis in ('x', 'y'):
                    try:
                        self.observers[axis].update(
                            self.payload_time,
                            float(self.cart_mm[axis]) / 1000.0,
                            self.cart_vel_mm_s[axis] / 1000.0,
                            math.radians(self.angle_deg[axis] or 0.0),
                            self.last_applied_cmd[axis] / 1000.0,
                        )
                    except ValueError as exc:
                        self.get_logger().warn(f'{axis}-axis observer update rejected: {exc}')

    def _joy_cb(self, msg: Joy):
        self.last_joy_wall = time.monotonic()
        if self.done or self.phase == 'arming':
            return
        if len(msg.buttons) > 9:
            pressed = bool(msg.buttons[9])
            if pressed and not self.last_mode_button:
                if self.phase == 'idle':
                    self.selected_mode = (
                        'pulse' if self.selected_mode == 'hybrid' else 'hybrid')
                    self.get_logger().info(
                        f'Jog mode -> {self.selected_mode.upper()} '
                        '(Button 10)')
                else:
                    self.get_logger().warn(
                        'Finish or release the current jog before changing Pulse/Hybrid mode')
            self.last_mode_button = pressed
        if len(msg.axes) < 2 or len(msg.buttons) < 6:
            return
        if msg.buttons[5]:
            self._request_estop()
            return
        lb = bool(msg.buttons[4])
        raw_x = -float(msg.axes[0])
        raw_y = float(msg.axes[1])
        if self.args.axis == 'x':
            raw_y = 0.0
        elif self.args.axis == 'y':
            raw_x = 0.0
        mag = math.hypot(raw_x, raw_y)
        active = lb and mag > self.args.deadzone
        if active:
            scale = (mag - self.args.deadzone) / max(1.0e-9, 1.0 - self.args.deadzone)
            scale = max(0.0, min(1.0, scale))
            direction = {'x': raw_x / mag if mag else 0.0, 'y': raw_y / mag if mag else 0.0}
        else:
            scale = 0.0
            direction = {'x': 0.0, 'y': 0.0}

        if active and self.phase == 'idle' and self.selected_mode == 'pulse':
            self.joy_dir = direction
            self.joy_magnitude = scale
            self.phase = 'pulse'
            self._publish_pulse()
            self.get_logger().info('Pulse jog started')
        elif active and self.phase == 'pulse':
            self.joy_dir = direction
            self.joy_magnitude = scale
            self._publish_pulse()
        elif not active and self.phase == 'pulse':
            self._publish(0.0, 0.0)
            self.phase = 'idle'
            self.get_logger().info('Pulse jog stopped')
        elif active and self.phase == 'idle':
            self._invalidate_stop_plans()
            self.joy_dir = direction
            # Match the existing ZV_JOG behavior: the stick selects direction
            # while the configured jog speed is fixed for the entire shaped
            # move.  A partially deflected stick must not make the front-end
            # imperceptibly slow or alter an impulse amplitude mid-sequence.
            self.joy_magnitude = 1.0
            self.phase = 'zvd_1'
            self.transition_wall = time.monotonic()
            self.get_logger().info(
                f'ZVD front end started dir=({direction["x"]:+.2f},{direction["y"]:+.2f})')
        elif active and self.phase in ('zvd_1', 'zvd_2', 'full_speed'):
            # Direction and magnitude are held from the first impulse; changing
            # either mid-shaper would invalidate the cancellation timing.
            pass
        elif not active and self.phase in ('zvd_1', 'zvd_2', 'full_speed'):
            release_stamp = float(msg.header.stamp.sec) + 1.0e-9 * float(msg.header.stamp.nanosec)
            self._capture_endpoint_and_plan(release_stamp)
        self.stick_active = active

    def _ready(self) -> bool:
        if self.latest_gantry is None or self.latest_gantry.estop or not self.latest_gantry.homed:
            return False
        if (
            self.gantry_wall is None
            or time.monotonic() - self.gantry_wall > self.args.state_fresh_timeout
        ):
            return False
        if (
            self.payload_wall is None
            or time.monotonic() - self.payload_wall > self.args.payload_fresh_timeout
        ):
            return False
        return all(self.cart_mm[a] is not None and self.angle_deg[a] is not None
                   for a in self.active_axes)

    def _arm_when_ready(self):
        if self.done or self.phase != 'arming':
            return
        if not self._ready():
            return
        if not self.mode_cli.wait_for_service(timeout_sec=0.1):
            return
        req = SetMode.Request()
        req.mode = 'TRAJ'
        future = self.mode_cli.call_async(req)
        future.add_done_callback(self._on_set_traj)
        self.phase = 'arm_pending'

    def _on_set_traj(self, future):
        try:
            result = future.result()
        except Exception as exc:
            self._stop(f'could not enter TRAJ: {exc}')
            return
        if result is None or not result.success:
            self._stop('could not enter TRAJ mode')
            return
        if self.latest_gantry is not None and self.latest_gantry.enabled:
            self.phase = 'idle'
            self.get_logger().info('TRAJ armed. Ready for hybrid jog input.')
            return
        if not self.enable_cli.wait_for_service(timeout_sec=1.0):
            self._stop('/gantry/enable unavailable')
            return
        future = self.enable_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_enable)

    def _on_enable(self, future):
        try:
            result = future.result()
        except Exception as exc:
            self._stop(f'enable failed: {exc}')
            return
        if result is None or not result.success:
            self._stop('could not enable TRAJ stream')
            return
        self.phase = 'idle'
        self.get_logger().info('TRAJ armed. Ready for hybrid jog input.')

    def _timer_cb(self):
        if self.done:
            return
        now = time.monotonic()
        if self.phase in (
            'pulse', 'zvd_1', 'zvd_2', 'full_speed',
            'planning', 'optimal_stop', 'zv_stop', 'verify_stop',
        ):
            safety_reason = self._runtime_safety_reason()
            if safety_reason is not None:
                self._stop(safety_reason)
                return
        if self.phase in ('pulse', 'zvd_1', 'zvd_2', 'full_speed') and (
            self.last_joy_wall is None
            or now - self.last_joy_wall > self.args.joy_fresh_timeout
        ):
            self._stop('joystick data stale during commanded motion')
            return
        if self.phase == 'pulse':
            self._publish_pulse()
        elif self.phase in ('zvd_1', 'zvd_2', 'full_speed'):
            if (
                self.payload_wall is None
                or now - self.payload_wall > self.args.payload_fresh_timeout
            ):
                self._stop('payload data stale during ZVD jog')
                return
            elapsed = now - (self.transition_wall or now)
            if self.phase == 'zvd_1' and elapsed >= self.zvd_T:
                self.phase = 'zvd_2'
                self.transition_wall = now
            elif self.phase == 'zvd_2' and elapsed >= self.zvd_T:
                self.phase = 'full_speed'
                self.get_logger().info('ZVD front end complete; full-speed jog active')
            self._publish_zvd_frontend()
            self._maybe_refresh_stop_plan(now)
        elif self.phase == 'planning':
            self._publish(0.0, 0.0)
            self._poll_optimal_plan(now)
        elif self.phase == 'optimal_stop':
            self._run_open_loop(now)
        elif self.phase == 'zv_stop':
            self._run_zv_stop(now)
        elif self.phase == 'verify_stop':
            self._publish(0.0, 0.0)
            self._verify_open_loop_terminal(now)

        if self.phase in ('planning', 'optimal_stop', 'zv_stop', 'verify_stop'):
            if (
                self.payload_wall is None
                or now - self.payload_wall > self.args.payload_fresh_timeout
            ):
                self._stop('payload data stale during shaped stop control')

    def _zvd_weights(self) -> tuple[float, float, float]:
        zeta = max(0.0, min(0.99, self.args.zvd_zeta))
        root = math.sqrt(max(1.0 - zeta * zeta, 1.0e-12))
        k = math.exp(-math.pi * zeta / root)
        denom = (1.0 + k) ** 2
        return 1.0 / denom, 2.0 * k / denom, k * k / denom

    def _zv_release_gain(self) -> float:
        """Remaining velocity fraction after the first impulse of a ZV stop."""
        zeta = max(0.0, min(0.99, self.args.zvd_zeta))
        root = math.sqrt(max(1.0 - zeta * zeta, 1.0e-12))
        k = math.exp(-math.pi * zeta / root)
        return k / (1.0 + k)

    def _publish_zvd_frontend(self):
        a0, a1, _ = self._zvd_weights()
        gain = a0 if self.phase == 'zvd_1' else (a0 + a1 if self.phase == 'zvd_2' else 1.0)
        speed = gain * self.args.jog_speed_mm_s * self.joy_magnitude
        self._publish(self.joy_dir['x'] * speed, self.joy_dir['y'] * speed)

    def _publish_pulse(self):
        speed = self.args.jog_speed_mm_s * self.joy_magnitude
        self._publish(self.joy_dir['x'] * speed, self.joy_dir['y'] * speed)

    def _invalidate_stop_plans(self):
        if self._plan_future is not None:
            self._plan_future.cancel()
        self._plan_future = None
        self._plan_future_meta = None
        self._cached_stop_plan = None
        self._cached_plan_meta = None
        self._last_plan_submit_wall = -math.inf

    def _stop_plan_snapshot(self, now: float) -> tuple[dict, dict] | None:
        """Predict the measured state to the expected LP completion instant."""
        if not self._ready():
            return None
        states = {}
        directions = {}
        initial_commands = {}
        forward_workspace = {}
        forecast_s = self._planning_forecast_s
        forecast_a, forecast_b = discrete_crane_model(
            self._forecast_model, forecast_s
        )
        for axis in self.active_axes:
            if abs(self.joy_dir[axis]) < 1.0e-3:
                continue
            observer_state = self.observers[axis].state
            if observer_state is None:
                return None
            direction = math.copysign(1.0, self.joy_dir[axis])
            state = np.asarray(observer_state, dtype=float).copy()
            command_m_s = self.last_applied_cmd[axis] / 1000.0
            forecast = (
                forecast_a @ state
                + forecast_b[:, 0] * command_m_s
            )
            forecast_cart_mm = 1000.0 * forecast[0]
            forecast[0] = 0.0  # Translation is irrelevant because final position is free.
            states[axis] = forecast.tolist()
            directions[axis] = direction
            initial_commands[axis] = command_m_s
            if direction > 0.0:
                available_mm = (
                    self.args.workspace_max_mm - self.args.workspace_margin_mm
                    - forecast_cart_mm
                )
            else:
                available_mm = (
                    forecast_cart_mm - self.args.workspace_min_mm
                    - self.args.workspace_margin_mm
                )
            available_mm = min(available_mm, self.args.max_forward_stop_distance_mm)
            if available_mm <= 1.0:
                return None
            forward_workspace[axis] = available_mm / 1000.0
        if not states:
            return None
        metadata = {
            'submit_wall': now,
            'forecast_wall': now + forecast_s,
            'states': states,
            'directions': directions,
            'initial_commands': initial_commands,
            'forward_workspace': forward_workspace,
        }
        arguments = {
            'direction_signs': directions,
            'initial_commands_m_s': initial_commands,
            'forward_workspace_m': forward_workspace,
        }
        return metadata, arguments

    def _consume_background_plan(self, now: float):
        if self._plan_future is None or not self._plan_future.done():
            return
        future = self._plan_future
        metadata = self._plan_future_meta
        self._plan_future = None
        self._plan_future_meta = None
        try:
            plan = future.result()
        except (OpenLoopPlanningError, ValueError, RuntimeError) as exc:
            if now - self._last_plan_report_wall >= 1.0:
                self._last_plan_report_wall = now
                self.get_logger().warn(f'Forward-only stop plan unavailable: {exc}')
            return
        except Exception as exc:
            if now - self._last_plan_report_wall >= 1.0:
                self._last_plan_report_wall = now
                self.get_logger().error(f'Unexpected stop-planner failure: {exc}')
            return
        self._cached_stop_plan = plan
        self._cached_plan_meta = metadata
        observed_wall = now - float(metadata['submit_wall'])
        target_forecast = observed_wall + 0.02
        self._planning_forecast_s = max(
            self.args.optimal_plan_forecast_s,
            min(
                self.args.optimal_max_plan_age_s,
                0.5 * self._planning_forecast_s + 0.5 * target_forecast,
            ),
        )
        if now - self._last_plan_report_wall >= 1.0:
            self._last_plan_report_wall = now
            self.get_logger().info(
                f'Stop-now plan refreshed: {plan.duration_s:.2f}s, '
                f'LP wall={plan.solve_time_s:.3f}s'
            )

    def _maybe_refresh_stop_plan(self, now: float):
        self._consume_background_plan(now)
        if self._plan_future is not None:
            return
        if now - self._last_plan_submit_wall < self.args.optimal_replan_period_s:
            return
        snapshot = self._stop_plan_snapshot(now)
        if snapshot is None:
            return
        metadata, arguments = snapshot
        self._last_plan_submit_wall = now
        self._plan_future_meta = metadata
        self._plan_future = self._planner_executor.submit(
            self.planner.plan,
            metadata['states'],
            **arguments,
        )

    def _cached_plan_is_fresh(self, now: float) -> tuple[bool, str]:
        plan = self._cached_stop_plan
        metadata = self._cached_plan_meta
        if plan is None or metadata is None:
            return False, 'no completed stop-now plan'
        age = abs(now - float(metadata['forecast_wall']))
        if age > self.args.optimal_max_plan_age_s:
            return False, f'cached plan is {age:.3f}s from its forecast state'
        for axis_index, axis in enumerate(plan.axes):
            observer_state = self.observers[axis].state
            if observer_state is None:
                return False, f'{axis}-axis observer is unavailable'
            direction = float(metadata['directions'][axis])
            current = np.array([
                0.0,
                float(observer_state[1]),
                math.radians(self.angle_deg[axis] or 0.0),
                float(observer_state[3]),
            ]) * direction
            expected = plan.initial_states[axis_index]
            if abs(current[1] - expected[1]) > (
                self.args.max_cached_velocity_error_mm_s / 1000.0
            ):
                return False, f'{axis}-axis velocity changed since the cached plan'
            if abs(current[2] - expected[2]) > math.radians(
                self.args.max_cached_angle_error_deg
            ):
                return False, f'{axis}-axis angle changed since the cached plan'
            if abs(current[3] - expected[3]) > math.radians(
                self.args.max_cached_angle_rate_error_deg_s
            ):
                return False, f'{axis}-axis angle rate changed since the cached plan'
        return True, ''

    def _capture_endpoint_and_plan(self, release_stamp: float):
        del release_stamp  # Final position is intentionally not the release coordinate.
        if not self._ready():
            self._stop('cannot start optimal stop: state or payload is unavailable')
            return
        now = time.monotonic()
        self._consume_background_plan(now)
        usable, reason = self._cached_plan_is_fresh(now)
        if not usable:
            self._start_zv_stop(now, reason)
            return

        plan = self._cached_stop_plan
        assert plan is not None
        self.open_loop_plan = plan
        self.plan_start_wall = now
        self.verification_start = None
        self.rms_window.clear()
        self._quiescent_since = None
        for axis in self.active_axes:
            self.plan_origin_cart_mm[axis] = float(self.cart_mm[axis])
            self.target_cart_mm[axis] = None
        nominal = plan.predicted_states[self.planner.nominal_scenario.label]
        for axis_index, axis in enumerate(plan.axes):
            signed_displacement_mm = 1000.0 * (
                nominal[-1, axis_index, 0] - nominal[0, axis_index, 0]
            )
            direction = plan.direction_signs[axis_index]
            self.target_cart_mm[axis] = (
                float(self.plan_origin_cart_mm[axis])
                + direction * signed_displacement_mm
            )
        self.phase = 'optimal_stop'
        if self._plan_future is not None:
            self._plan_future.cancel()
        self._plan_future = None
        self._plan_future_meta = None
        target_text = ','.join(
            f'{axis}={self.target_cart_mm[axis]:.1f}mm' for axis in plan.axes
        )
        self.get_logger().info(
            f'Release: playing fresh {plan.duration_s:.2f}s forward-only '
            f'minimum-time plan; predicted stop {target_text}'
        )
        self._run_open_loop(now)

    def _start_zv_stop(self, now: float, optimal_reason: str):
        """Use the deterministic one-switch stop when no optimal plan is ready."""
        gain = self._zv_release_gain()
        command = {
            axis: gain * float(self.last_applied_cmd[axis])
            for axis in ('x', 'y')
        }
        for axis in self.active_axes:
            cart_mm = float(self.cart_mm[axis])
            travel_mm = abs(command[axis]) * self.zvd_T
            if command[axis] >= 0.0:
                workspace_mm = (
                    self.args.workspace_max_mm - self.args.workspace_margin_mm
                    - cart_mm
                )
            else:
                workspace_mm = (
                    cart_mm - self.args.workspace_min_mm
                    - self.args.workspace_margin_mm
                )
            available_mm = min(
                workspace_mm, self.args.max_forward_stop_distance_mm
            )
            if travel_mm > max(0.0, available_mm):
                self._return_to_idle(
                    f'optimal plan unavailable ({optimal_reason}) and the ZV '
                    f'fallback needs {travel_mm:.1f}mm on {axis}, but only '
                    f'{max(0.0, available_mm):.1f}mm is available'
                )
                return
            self.plan_origin_cart_mm[axis] = cart_mm
            self.target_cart_mm[axis] = cart_mm + command[axis] * self.zvd_T

        self.zv_stop_cmd = command
        self.zv_stop_reason = optimal_reason
        self.open_loop_plan = None
        self.plan_start_wall = now
        self.transition_wall = now
        self.verification_start = None
        self._quiescent_since = None
        self.rms_window.clear()
        self.phase = 'zv_stop'
        self._invalidate_stop_plans()
        self.get_logger().warn(
            f'Optimal stop unavailable ({optimal_reason}); applying ZV release '
            f'tail for {self.zvd_T:.3f}s at {gain:.3f} of release speed'
        )
        self._publish(command['x'], command['y'])

    def _run_zv_stop(self, now: float):
        if self.phase != 'zv_stop':
            return
        elapsed = now - (self.transition_wall or now)
        if elapsed < self.zvd_T:
            self._publish(self.zv_stop_cmd['x'], self.zv_stop_cmd['y'])
            errors = {'x': 0.0, 'y': 0.0}
            for axis in self.active_axes:
                errors[axis] = (
                    float(self.cart_mm[axis]) - float(self.target_cart_mm[axis])
                )
            angle_mag = math.hypot(*(
                self.angle_deg[axis] or 0.0 for axis in self.active_axes
            ))
            self._log(errors, self.zv_stop_cmd, angle_mag, 0.0, 0.0)
            return

        self._publish(0.0, 0.0)
        for axis in self.active_axes:
            self.target_cart_mm[axis] = float(self.cart_mm[axis])
        self.phase = 'verify_stop'
        self.verification_start = now
        self._quiescent_since = None
        self.rms_window.clear()
        self.get_logger().info(
            'ZV release tail complete; verifying measured terminal rest'
        )

    def _poll_optimal_plan(self, now: float):
        # Kept as a fail-safe for an old serialized phase value; all planning
        # now happens continuously while the stick is held.
        self._publish(0.0, 0.0)
        self._stop('entered obsolete release-time planning phase')

    def _runtime_safety_reason(self) -> str | None:
        if self.latest_gantry is None:
            return 'gantry state unavailable'
        if (
            self.gantry_wall is None
            or time.monotonic() - self.gantry_wall > self.args.state_fresh_timeout
        ):
            return 'gantry state stale'
        if self.latest_gantry.estop:
            return 'gantry E-stop active'
        if not self.latest_gantry.homed:
            return 'gantry is no longer homed'
        if not self.latest_gantry.enabled:
            return 'motors disabled during shaped stop'
        if self.latest_gantry.mode != 'TRAJ':
            return f'gantry mode changed to {self.latest_gantry.mode}'
        return None

    def _run_open_loop(self, now: float):
        if self.phase != 'optimal_stop' or self.open_loop_plan is None:
            return
        safety_reason = self._runtime_safety_reason()
        if safety_reason is not None:
            self._stop(safety_reason)
            return
        elapsed = now - (self.plan_start_wall or now)
        index = int(elapsed / self.open_loop_plan.sample_period_s)
        if index >= len(self.open_loop_plan.commands_m_s):
            self._publish(0.0, 0.0)
            self.phase = 'verify_stop'
            self.verification_start = now
            self._quiescent_since = None
            self.rms_window.clear()
            self.get_logger().info('Open-loop command complete; verifying measured terminal rest')
            return

        command = {'x': 0.0, 'y': 0.0}
        for axis_index, axis in enumerate(self.open_loop_plan.axes):
            command[axis] = 1000.0 * float(
                self.open_loop_plan.commands_m_s[index, axis_index]
            )
        self._publish(command['x'], command['y'])

        errors = {'x': 0.0, 'y': 0.0}
        errors.update({
            axis: float(self.cart_mm[axis]) - float(self.target_cart_mm[axis])
            for axis in self.open_loop_plan.axes
        })
        angle_mag = math.hypot(*(
            self.angle_deg[a] or 0.0 for a in self.open_loop_plan.axes
        ))
        self._log(errors, command, angle_mag, 0.0, 0.0)

        predicted = self.open_loop_plan.predicted_states[
            self.planner.nominal_scenario.label
        ][min(index, len(self.open_loop_plan.commands_m_s))]
        for axis_index, axis in enumerate(self.open_loop_plan.axes):
            direction = self.open_loop_plan.direction_signs[axis_index]
            signed_displacement_m = direction * (
                float(self.cart_mm[axis]) - float(self.plan_origin_cart_mm[axis])
            ) / 1000.0
            signed_angle_rad = direction * math.radians(self.angle_deg[axis] or 0.0)
            position_deviation_mm = 1000.0 * abs(
                signed_displacement_m - predicted[axis_index, 0]
            )
            angle_deviation_deg = math.degrees(abs(
                signed_angle_rad - predicted[axis_index, 2]
            ))
            if position_deviation_mm > self.args.max_model_deviation_position_mm:
                self._start_zv_stop(
                    now,
                    f'{axis}-axis open-loop position/model deviation '
                    f'{position_deviation_mm:.1f}mm exceeds limit'
                )
                return
            if angle_deviation_deg > self.args.max_model_deviation_angle_deg:
                self._start_zv_stop(
                    now,
                    f'{axis}-axis open-loop angle/model deviation '
                    f'{angle_deviation_deg:.2f}deg exceeds limit'
                )
                return

        if now - self.last_status_wall >= self.args.print_period:
            self.last_status_wall = now
            self.get_logger().info(
                f'OPT t={elapsed:.2f}s sample={index}/'
                f'{len(self.open_loop_plan.commands_m_s)} '
                f'predicted_stop_error=({errors.get("x", 0.0):+.2f},'
                f'{errors.get("y", 0.0):+.2f})mm '
                f'cmd=({command["x"]:+.1f},{command["y"]:+.1f})mm/s')

    def _verify_open_loop_terminal(self, now: float):
        if self.phase != 'verify_stop' or not self._ready():
            return
        safety_reason = self._runtime_safety_reason()
        if safety_reason is not None:
            self._stop(safety_reason)
            return
        axes = self.open_loop_plan.axes if self.open_loop_plan is not None else self.active_axes
        errors = {'x': 0.0, 'y': 0.0}
        errors.update({
            axis: float(self.cart_mm[axis]) - float(self.target_cart_mm[axis])
            for axis in axes
        })
        angle_mag = math.hypot(*(self.angle_deg[a] or 0.0 for a in axes))
        predicted_stop_error = math.hypot(*(errors[a] for a in axes))
        angle_rate_mag = math.hypot(*(
            math.degrees(float(self.observers[a].state[3]))
            for a in axes
        ))
        cart_velocity_mag = math.hypot(*(self.cart_vel_mm_s[a] for a in axes))
        verify_t = now - (self.verification_start or now)
        self.rms_window.append((verify_t, angle_mag))
        while self.rms_window and self.rms_window[0][0] < verify_t - self.args.done_window_s:
            self.rms_window.popleft()
        values = [value for _, value in self.rms_window]
        rms = math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0
        peak = max(map(abs, values)) if values else 0.0
        quiet = (
            angle_mag <= self.args.quiescent_angle_deg
            and angle_rate_mag <= self.args.quiescent_angle_rate_deg_s
            and cart_velocity_mag <= self.args.quiescent_cart_vel_mm_s
        )
        if quiet:
            if self._quiescent_since is None:
                self._quiescent_since = now
        else:
            self._quiescent_since = None
        self._log(errors, {'x': 0.0, 'y': 0.0}, angle_mag, rms, peak)
        if now - self.last_status_wall >= self.args.print_period:
            self.last_status_wall = now
            self.get_logger().info(
                f'VERIFY t={verify_t:.2f}s predicted_stop_error='
                f'{predicted_stop_error:.2f}mm '
                f'angle={angle_mag:.3f}deg rate={angle_rate_mag:.2f}deg/s '
                f'cart_speed={cart_velocity_mag:.2f}mm/s')
        if (
            self._quiescent_since is not None
            and now - self._quiescent_since >= self.args.quiescent_hold_s
        ):
            self._complete_hybrid_move()
            return
        if verify_t >= self.args.open_loop_verify_timeout_s:
            self._return_to_idle(
                'open-loop command ended outside measured terminal tolerances; '
                'feedback correction is intentionally disabled'
            )

    def _clear_completed_move(self):
        """Clear per-move state while keeping the joystick controller armed."""
        self.phase = 'idle'
        self.transition_wall = None
        self.target_cart_mm = {'x': None, 'y': None}
        self.plan_origin_cart_mm = {'x': None, 'y': None}
        self.open_loop_plan = None
        self.plan_start_wall = None
        self.verification_start = None
        self._quiescent_since = None
        self.rms_window.clear()
        self.joy_dir = {'x': 0.0, 'y': 0.0}
        self.joy_magnitude = 0.0
        self.stick_active = False
        self.zv_stop_cmd = {'x': 0.0, 'y': 0.0}
        self.zv_stop_reason = ''
        self._invalidate_stop_plans()

    def _return_to_idle(self, reason: str):
        """Fail one jog safely without terminating the repeatable controller."""
        self._publish(0.0, 0.0)
        if self.csv_file is not None:
            self.csv_file.flush()
        self._clear_completed_move()
        self.get_logger().warn(
            f'Hybrid jog stopped: {reason}. Controller remains armed; '
            'release the stick, then start the next jog when the payload is safe.'
        )

    def _complete_hybrid_move(self):
        self._publish(0.0, 0.0)
        if self.csv_file is not None:
            self.csv_file.flush()
        self._clear_completed_move()
        self.get_logger().info(
            'Hybrid move complete: forward-only stop and measured payload settled. '
            f'Jog mode remains {self.selected_mode.upper()}.')

    def _publish(self, vx: float, vy: float):
        self.last_applied_cmd['x'] = float(vx)
        self.last_applied_cmd['y'] = float(vy)
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = TrajCmd.STREAM
        msg.vx_mm_s, msg.vy_mm_s = float(vx), float(vy)
        if self.latest_gantry is not None:
            msg.x, msg.y = float(self.latest_gantry.x), float(self.latest_gantry.y)
        self.traj_pub.publish(msg)

    def _log(self, errors, cmd, angle_mag, rms, peak):
        if self.csv_writer is None:
            return
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds * 1e-9, self.phase,
            self.target_cart_mm['x'], self.target_cart_mm['y'],
            self.cart_mm['x'], self.cart_mm['y'],
            errors['x'], errors['y'], self.angle_deg['x'], self.angle_deg['y'], angle_mag,
            cmd['x'], cmd['y'], rms, peak,
        ])
        if self.csv_file is not None and time.monotonic() - self._last_csv_flush_wall >= 0.5:
            self.csv_file.flush()
            self._last_csv_flush_wall = time.monotonic()

    def _request_estop(self):
        self._publish(0.0, 0.0)
        if self.estop_cli.wait_for_service(timeout_sec=0.2):
            self.estop_cli.call_async(Trigger.Request())
        self._stop('RB pressed: E-stop requested')

    def _stop(self, reason: str):
        if self.done:
            return
        self._publish(0.0, 0.0)
        if self.csv_file is not None:
            self.csv_file.flush()
        if self._plan_future is not None:
            self._plan_future.cancel()
        self.get_logger().info(f'Hybrid ZVD-optimal jog finished: {reason}')
        self.done = True

    def destroy_node(self):
        try:
            self._publish(0.0, 0.0)
        except Exception:
            pass
        if self.csv_file is not None:
            self.csv_file.close()
        self._planner_executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis', choices=('x', 'y', 'both'), default='both')
    p.add_argument('--start-mode', choices=('pulse', 'hybrid'), default='hybrid')
    p.add_argument('--payload-topic', default='/payload/pose_e_rel')
    p.add_argument('--rope-length-m', type=float, default=0.90)
    p.add_argument('--zvd-t-s', type=float, default=0.0,
                   help='ZVD impulse spacing; <=0 uses pi*sqrt(L/g).')
    p.add_argument('--zvd-zeta', type=float, default=0.0)
    p.add_argument('--jog-speed-mm-s', type=float, default=100.0)
    p.add_argument('--max-regulate-vel-mm-s', type=float, default=25.0,
                   help=argparse.SUPPRESS)
    p.add_argument('--max-regulate-accel-mm-s2', type=float, default=1000.0,
                   help='Vector command slew limit during the minimum-time stop.')
    p.add_argument('--max-correction-excursion-mm', type=float, default=80.0,
                   help=argparse.SUPPRESS)
    p.add_argument('--max-forward-stop-distance-mm', type=float, default=250.0,
                   help='Maximum allowed forward travel after joystick release.')
    p.add_argument(
        '--final-cart-tolerance-mm', type=float, default=10.0,
        help=argparse.SUPPRESS,
    )
    p.add_argument('--done-window-s', type=float, default=1.5)
    p.add_argument('--quiescent-angle-deg', type=float, default=0.12)
    p.add_argument('--quiescent-angle-rate-deg-s', type=float, default=0.50)
    p.add_argument('--quiescent-cart-vel-mm-s', type=float, default=2.0)
    p.add_argument('--quiescent-hold-s', type=float, default=0.40)
    p.add_argument('--open-loop-verify-timeout-s', type=float, default=2.0)
    p.add_argument('--rate-hz', type=float, default=100.0)
    p.add_argument('--payload-fresh-timeout', type=float, default=0.25)
    p.add_argument('--joy-fresh-timeout', type=float, default=0.25)
    p.add_argument('--state-fresh-timeout', type=float, default=0.25)
    p.add_argument('--deadzone', type=float, default=0.08)
    p.add_argument('--model-zeta', type=float, default=0.02)
    p.add_argument('--actuator-tau-s', type=float, default=0.045)
    p.add_argument('--angle-measurement-std-deg', type=float, default=0.06)
    p.add_argument('--optimal-sample-period-s', type=float, default=0.02)
    p.add_argument('--optimal-control-knot-period-s', type=float, default=0.10,
                   help=argparse.SUPPRESS)
    p.add_argument('--optimal-min-horizon-s', type=float, default=0.30)
    p.add_argument('--optimal-horizon-s', type=float, default=2.80,
                   help='Maximum forward-only stopping horizon.')
    p.add_argument('--optimal-horizon-resolution-s', type=float, default=0.04)
    p.add_argument('--optimal-lead-time-s', type=float, default=1.0,
                   help=argparse.SUPPRESS)
    p.add_argument('--optimal-plan-forecast-s', type=float, default=0.12,
                   help='Forecast used to align plans with their expected completion time.')
    p.add_argument('--optimal-replan-period-s', type=float, default=0.10)
    p.add_argument('--optimal-max-plan-age-s', type=float, default=0.50)
    p.add_argument('--max-cached-velocity-error-mm-s', type=float, default=20.0)
    p.add_argument('--max-cached-angle-error-deg', type=float, default=0.25)
    p.add_argument('--max-cached-angle-rate-error-deg-s', type=float, default=1.0)
    p.add_argument('--rope-length-uncertainty-fraction', type=float, default=0.10)
    p.add_argument('--actuator-uncertainty-fraction', type=float, default=0.15)
    p.add_argument('--uncertainty-samples', type=int, choices=(1, 3), default=3)
    p.add_argument('--robust-terminal-multiplier', type=float, default=2.0)
    p.add_argument('--optimizer-time-limit-s', type=float, default=0.08)
    p.add_argument('--planner-wall-timeout-s', type=float, default=1.5,
                   help=argparse.SUPPRESS)
    p.add_argument('--planner-lead-margin-s', type=float, default=0.08,
                   help=argparse.SUPPRESS)
    p.add_argument('--workspace-min-mm', type=float, default=0.0)
    p.add_argument('--workspace-max-mm', type=float, default=1150.0)
    p.add_argument('--workspace-margin-mm', type=float, default=5.0)
    p.add_argument('--max-model-deviation-position-mm', type=float, default=20.0)
    p.add_argument('--max-model-deviation-angle-deg', type=float, default=1.0)
    p.add_argument('--print-period', type=float, default=0.25)
    p.add_argument('--log-csv', default='')
    return p.parse_args()


def main():
    args = parse_args()
    positive = (
        args.rope_length_m,
        args.rate_hz,
        args.jog_speed_mm_s,
        args.max_regulate_vel_mm_s,
        args.max_regulate_accel_mm_s2,
        args.max_forward_stop_distance_mm,
        args.optimal_sample_period_s,
        args.optimal_min_horizon_s,
        args.optimal_horizon_s,
        args.optimal_horizon_resolution_s,
        args.optimal_plan_forecast_s,
        args.optimal_replan_period_s,
        args.optimal_max_plan_age_s,
        args.max_cached_velocity_error_mm_s,
        args.max_cached_angle_error_deg,
        args.max_cached_angle_rate_error_deg_s,
        args.payload_fresh_timeout,
        args.joy_fresh_timeout,
        args.state_fresh_timeout,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        print('Invalid positive physical/rate argument.', file=sys.stderr)
        return 2
    if args.optimal_horizon_s <= args.optimal_min_horizon_s:
        print('Optimal maximum horizon must exceed its minimum.', file=sys.stderr)
        return 2
    if args.workspace_max_mm <= args.workspace_min_mm + 2.0 * args.workspace_margin_mm:
        print('Workspace bounds/margin leave no usable interval.', file=sys.stderr)
        return 2
    rclpy.init()
    node = HybridZvdOptimalJog(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

