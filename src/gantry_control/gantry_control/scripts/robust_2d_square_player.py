#!/usr/bin/env python3
"""Run a robust ZVD-shaped +X, +Y, -X, -Y square trajectory."""

from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime
import math
from pathlib import Path
import sys
import time

import rclpy
from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import SetMode
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from robust_2d_shaper import (
    Robust2dSquareProfile,
    SwayEstimate,
    SwayState,
    estimate_sway_state,
    propagate_sway_state,
)


class Robust2dSquarePlayer(Node):
    def __init__(self, args: argparse.Namespace, profile: Robust2dSquareProfile):
        super().__init__('robust_2d_square_player')
        self.args = args
        self.base_profile = profile
        self.profile = profile
        self.phase = 'waiting'
        self.done = False
        self.success = False
        self.latest_gantry: GantryState | None = None
        self.gantry_wall: float | None = None
        self.payload_wall: float | None = None
        self.payload_time = float('nan')
        self.pitch_deg = float('nan')
        self.roll_deg = float('nan')
        self.payload_history: deque[tuple[float, float, float]] = deque(maxlen=5000)
        self.pitch_estimate: SwayEstimate | None = None
        self.roll_estimate: SwayEstimate | None = None
        self.planned_pitch_state: SwayState | None = None
        self.planned_roll_state: SwayState | None = None
        self.ic_planned_start_wall: float | None = None
        self.last_ic_plan_attempt_wall = -math.inf
        self.last_ic_validation_wall = -math.inf
        self.y_nzic_ready = False
        self.y_nzic_started = False
        self.y_hold_start_wall: float | None = None
        self.last_y_plan_attempt_wall = -math.inf
        self.last_y_validation_wall = -math.inf
        self.origin_x_mm: float | None = None
        self.origin_y_mm: float | None = None
        self.motion_start_wall: float | None = None
        self.preflight_start_wall: float | None = None
        self.quiescent_since_wall: float | None = None
        self.last_status_wall = -math.inf
        self.last_csv_flush_wall = time.monotonic()
        self.settle_samples: list[tuple[float, float]] = []
        self.service_missing_since_wall: float | None = None
        self._mode_future = None
        self._enable_future = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.traj_pub = self.create_publisher(TrajCmd, '/traj_cmd', qos)
        self.create_subscription(GantryState, '/gantry/state', self._gantry_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            args.payload_topic,
            self._payload_cb,
            qos_profile_sensor_data,
        )
        self.mode_client = self.create_client(SetMode, '/gantry/set_mode')
        self.enable_client = self.create_client(Trigger, '/gantry/enable')

        self.csv_file = None
        self.csv_writer = None
        self.log_path: Path | None = None
        self._open_log(args.log_csv)

        self.arm_timer = self.create_timer(0.2, self._arm_when_ready)
        self.command_timer = self.create_timer(1.0 / args.rate_hz, self._tick)
        mode_summary = ', '.join(
            f'{mode.name}={mode.natural_frequency_hz:.3f}Hz'
            for mode in profile.modes
        )
        self.get_logger().info(
            '2-D robust square player waiting for fresh gantry/payload data: '
            f'+X,+Y,-X,-Y; {profile.x_distance_mm:.1f} x '
            f'{profile.y_distance_mm:.1f} mm at {profile.speed_mm_s:.1f} mm/s; '
            f'{mode_summary}, {len(profile.amplitudes)} impulses, '
            f'tail={profile.shaper_tail_s:.4f}s.'
        )

    def _open_log(self, requested_path: str) -> None:
        if not requested_path:
            return
        if requested_path == 'auto':
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = Path.cwd() / 'logs' / 'robust_2d' / f'square_{stamp}.csv'
        else:
            path = Path(requested_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = path.resolve()
        self.csv_file = path.open('w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'wall_time_s', 'motion_time_s', 'phase',
            'cmd_vx_mm_s', 'cmd_vy_mm_s',
            'cart_x_mm', 'cart_y_mm', 'relative_x_mm', 'relative_y_mm',
            'expected_relative_x_mm', 'expected_relative_y_mm',
            'path_error_x_mm', 'path_error_y_mm',
            'cart_vx_mm_s', 'cart_vy_mm_s',
            'payload_time_s', 'pitch_deg', 'roll_deg', 'angle_magnitude_deg',
            'nzic_enabled',
            'initial_pitch_deg', 'initial_pitch_rate_deg_s',
            'initial_roll_deg', 'initial_roll_rate_deg_s',
            'pitch_fit_rmse_deg', 'roll_fit_rmse_deg',
            'nzic_correction_count',
        ])

    def _gantry_cb(self, msg: GantryState) -> None:
        self.latest_gantry = msg
        self.gantry_wall = time.monotonic()

    def _payload_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 3:
            return
        self.payload_time = float(msg.data[0])
        self.pitch_deg = float(msg.data[1])
        self.roll_deg = float(msg.data[2])
        self.payload_wall = time.monotonic()
        if math.isfinite(self.pitch_deg) and math.isfinite(self.roll_deg):
            self.payload_history.append((
                self.payload_wall,
                math.radians(self.pitch_deg),
                math.radians(self.roll_deg),
            ))

    def _fresh(self, stamp: float | None, timeout_s: float) -> bool:
        return stamp is not None and time.monotonic() - stamp <= timeout_s

    def _arm_when_ready(self) -> None:
        if self.done or self.phase != 'waiting':
            return
        if not self._fresh(self.gantry_wall, self.args.state_fresh_timeout_s):
            self._periodic_info('Waiting for fresh /gantry/state')
            return
        if self.args.require_payload and not self._fresh(
            self.payload_wall, self.args.payload_fresh_timeout_s
        ):
            self._periodic_info(f'Waiting for fresh {self.args.payload_topic}')
            return
        assert self.latest_gantry is not None
        if self.latest_gantry.estop:
            self._periodic_info('Waiting: gantry E-stop is active')
            return
        if not self.latest_gantry.homed:
            self._periodic_info('Waiting: gantry must be homed')
            return
        if not self._workspace_allows_square(self.latest_gantry):
            self._stop(
                'square does not fit inside configured workspace from the current position'
            )
            return
        if not self._required_services_are_ready():
            return

        self.phase = 'arming'
        self.preflight_start_wall = time.monotonic()
        if self.latest_gantry.mode == 'TRAJ':
            self._request_enable()
            return
        request = SetMode.Request()
        request.mode = 'TRAJ'
        self._mode_future = self.mode_client.call_async(request)
        self._mode_future.add_done_callback(self._mode_response)

    def _mode_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # ROS future propagates service failures here.
            self._stop(f'failed to set TRAJ mode: {exc}')
            return
        if response is None or not response.success:
            message = '' if response is None else response.message
            self._stop(f'gantry rejected TRAJ mode: {message}')
            return
        self._request_enable()

    def _request_enable(self) -> None:
        if self.latest_gantry is not None and self.latest_gantry.enabled:
            self._begin_preflight()
            return
        if not self.enable_client.service_is_ready():
            # DDS discovery can briefly lose a service between the readiness
            # check and this callback. Return to the guarded waiting state and
            # retry instead of turning a transient into an experiment failure.
            self.phase = 'waiting'
            self.service_missing_since_wall = time.monotonic()
            self._periodic_info('Waiting for /gantry/enable service discovery')
            return
        self._enable_future = self.enable_client.call_async(Trigger.Request())
        self._enable_future.add_done_callback(self._enable_response)

    def _required_services_are_ready(self) -> bool:
        assert self.latest_gantry is not None
        missing: list[str] = []
        if (
            self.latest_gantry.mode != 'TRAJ'
            and not self.mode_client.service_is_ready()
        ):
            missing.append('/gantry/set_mode')
        if not self.latest_gantry.enabled and not self.enable_client.service_is_ready():
            missing.append('/gantry/enable')
        if not missing:
            self.service_missing_since_wall = None
            return True

        now = time.monotonic()
        if self.service_missing_since_wall is None:
            self.service_missing_since_wall = now
        elapsed_s = now - self.service_missing_since_wall
        if elapsed_s >= self.args.service_wait_timeout_s:
            self._stop(
                f'services unavailable for {elapsed_s:.1f}s: {", ".join(missing)}; '
                'verify mission_planner.launch.py is running in the same ROS domain'
            )
            return False
        self._periodic_info(
            f'Waiting for ROS service discovery ({elapsed_s:.1f}/'
            f'{self.args.service_wait_timeout_s:.1f}s): {", ".join(missing)}'
        )
        return False

    def _enable_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self._stop(f'failed to enable gantry: {exc}')
            return
        if response is None or not response.success:
            message = '' if response is None else response.message
            self._stop(f'gantry enable failed: {message}')
            return
        self._begin_preflight()

    def _begin_preflight(self) -> None:
        if self.done:
            return
        self.phase = 'preflight'
        self.preflight_start_wall = time.monotonic()
        if self.args.nonzero_ic:
            self.get_logger().info(
                'TRAJ requested and enable accepted; holding the cart at zero '
                'while the live encoder history estimates pitch/roll state and '
                'the monotonic robust NZIC profile is planned.'
            )
        else:
            self.get_logger().info(
                'TRAJ requested and enable accepted; holding zero until the '
                f'{self.args.start_hold_s:.2f} s measured-rest gate passes.'
            )

    def _preflight_ready(self, now: float) -> bool:
        self._publish(0.0, 0.0, 0.0)
        if self.preflight_start_wall is None:
            self.preflight_start_wall = now
        if now - self.preflight_start_wall > self.args.arm_timeout_s:
            self._stop('gantry did not reach a motion-ready TRAJ state before timeout')
            return False
        if not self._motion_state_is_safe(report=False):
            self.quiescent_since_wall = None
            return False
        assert self.latest_gantry is not None
        cart_speed_mm_s = 1000.0 * math.hypot(
            float(self.latest_gantry.vx), float(self.latest_gantry.vy)
        )
        angle_magnitude_deg = math.hypot(self.pitch_deg, self.roll_deg)
        cart_is_quiet = cart_speed_mm_s <= self.args.start_cart_speed_limit_mm_s
        payload_is_quiet = (
            self.args.nonzero_ic
            or not self.args.require_payload
            or angle_magnitude_deg <= self.args.start_angle_limit_deg
        )
        if not cart_is_quiet or not payload_is_quiet:
            self.quiescent_since_wall = None
            self._periodic_info(
                'Preflight rest gate: '
                f'cart={cart_speed_mm_s:.2f}/'
                f'{self.args.start_cart_speed_limit_mm_s:.2f}mm/s, '
                f'angle={angle_magnitude_deg:.3f}/'
                f'{self.args.start_angle_limit_deg:.3f}deg'
            )
            return False
        if self.quiescent_since_wall is None:
            self.quiescent_since_wall = now
        quiet_duration = now - self.quiescent_since_wall
        if (
            now - self.preflight_start_wall < self.args.start_delay_s
            or quiet_duration < self.args.start_hold_s
        ):
            return False
        if self.args.nonzero_ic:
            return self._nzic_preflight(now)
        return self._start_motion(now)

    def _estimate_sway(
        self,
        now: float,
        *,
        required_axes: tuple[str, ...] = ('pitch', 'roll'),
    ) -> tuple[SwayEstimate, SwayEstimate]:
        primary_mode = self.base_profile.modes[0]
        pitch_samples = ((stamp, pitch) for stamp, pitch, _ in self.payload_history)
        roll_samples = ((stamp, roll) for stamp, _, roll in self.payload_history)
        estimate_arguments = {
            'mode': primary_mode,
            'reference_time_s': now,
            'window_s': self.args.ic_estimator_window_s,
            'minimum_samples': self.args.ic_estimator_min_samples,
            'minimum_span_s': self.args.ic_estimator_min_span_s,
        }
        pitch = estimate_sway_state(pitch_samples, **estimate_arguments)
        roll = estimate_sway_state(roll_samples, **estimate_arguments)
        self.pitch_estimate = pitch
        self.roll_estimate = roll
        maximum_rmse = math.radians(self.args.ic_max_fit_rmse_deg)
        bad_fit = (
            ('pitch' in required_axes and pitch.rmse_rad > maximum_rmse)
            or ('roll' in required_axes and roll.rmse_rad > maximum_rmse)
        )
        if bad_fit:
            raise ValueError(
                'harmonic fit is too noisy: '
                f'pitch RMSE={math.degrees(pitch.rmse_rad):.4f}deg, '
                f'roll RMSE={math.degrees(roll.rmse_rad):.4f}deg, '
                f'limit={self.args.ic_max_fit_rmse_deg:.4f}deg'
            )
        maximum_amplitude = math.radians(self.args.ic_max_amplitude_deg)
        pitch_amplitude = pitch.state.amplitude_rad(primary_mode)
        roll_amplitude = roll.state.amplitude_rad(primary_mode)
        excessive_sway = (
            ('pitch' in required_axes and pitch_amplitude > maximum_amplitude)
            or ('roll' in required_axes and roll_amplitude > maximum_amplitude)
        )
        if excessive_sway:
            raise ValueError(
                'estimated initial sway exceeds the NZIC safety envelope: '
                f'pitch={math.degrees(pitch_amplitude):.3f}deg, '
                f'roll={math.degrees(roll_amplitude):.3f}deg, '
                f'limit={self.args.ic_max_amplitude_deg:.3f}deg'
            )
        return pitch, roll

    def _clear_nzic_plan(self) -> None:
        self.profile = self.base_profile
        self.ic_planned_start_wall = None
        self.planned_pitch_state = None
        self.planned_roll_state = None

    def _nzic_preflight(self, now: float) -> bool:
        primary_mode = self.base_profile.modes[0]
        if self.ic_planned_start_wall is not None:
            if now >= self.ic_planned_start_wall:
                return self._start_motion(now)
            if now - self.last_ic_validation_wall >= self.args.ic_replan_period_s:
                self.last_ic_validation_wall = now
                try:
                    pitch, _ = self._estimate_sway(now)
                except ValueError as exc:
                    self._clear_nzic_plan()
                    self._periodic_info(f'NZIC estimate rejected: {exc}')
                    return False
                remaining_s = self.ic_planned_start_wall - now
                forecast_pitch = propagate_sway_state(
                    pitch.state, primary_mode, remaining_s
                )
                assert self.planned_pitch_state is not None
                pitch_error = SwayState(
                    forecast_pitch.angle_rad - self.planned_pitch_state.angle_rad,
                    forecast_pitch.angular_rate_rad_s
                    - self.planned_pitch_state.angular_rate_rad_s,
                ).amplitude_rad(primary_mode)
                if pitch_error > math.radians(
                    self.args.ic_plan_validation_tolerance_deg
                ):
                    self._clear_nzic_plan()
                    self._periodic_info(
                        'NZIC state forecast changed; replanning '
                        f'(pitch error={math.degrees(pitch_error):.3f}deg)'
                    )
                    return False
            return False

        if now - self.last_ic_plan_attempt_wall < self.args.ic_replan_period_s:
            return False
        self.last_ic_plan_attempt_wall = now
        try:
            pitch, roll = self._estimate_sway(now)
        except ValueError as exc:
            self._periodic_info(f'NZIC estimator waiting: {exc}')
            return False

        planned_start = now + self.args.ic_planning_lead_s
        pitch_state = propagate_sway_state(
            pitch.state, primary_mode, planned_start - pitch.reference_time_s
        )
        roll_state = propagate_sway_state(
            roll.state, primary_mode, planned_start - roll.reference_time_s
        )
        try:
            candidate = self.base_profile.with_leg_nonzero_initial_condition(
                leg_index=0,
                initial_state=pitch_state,
                initial_state_time_s=0.0,
                minimum_correction_amplitude_rad=math.radians(
                    self.args.ic_min_amplitude_deg
                ),
                primary_band_fraction=self.args.ic_primary_band_fraction,
                second_mode_band_fraction=self.args.ic_second_band_fraction,
                maximum_second_mode_residual_fraction=(
                    self.args.ic_max_second_residual_fraction
                ),
            )
        except ValueError as exc:
            self._periodic_info(
                f'NZIC phase not yet feasible; holding zero and retrying: {exc}'
            )
            return False
        if time.monotonic() >= planned_start - 0.05:
            self._periodic_info(
                'NZIC solve missed its forecast window; replanning from fresh data'
            )
            return False

        self.profile = candidate
        self.planned_pitch_state = pitch_state
        self.planned_roll_state = roll_state
        self.ic_planned_start_wall = planned_start
        self.last_ic_validation_wall = now
        correction_summary = 'standard ZVD (estimated sway below correction threshold)'
        if candidate.ic_corrections:
            parts = []
            for correction in candidate.ic_corrections:
                parts.append(
                    f'{correction.leg_name}: A0='
                    f'{math.degrees(correction.initial_amplitude_rad):.3f}deg, '
                    f'nominal={math.degrees(correction.nominal_residual_rad):.3e}deg, '
                    f'rope-band={math.degrees(correction.primary_band_residual_rad):.3f}deg, '
                    f'high-band={correction.second_mode_band_residual_fraction:.4f}'
                )
            correction_summary = '; '.join(parts)
        self.get_logger().info(
            'NZIC plan accepted; continuing live state validation until start in '
            f'{planned_start - time.monotonic():.3f}s; {correction_summary}'
        )
        return False

    def _start_motion(self, now: float) -> bool:
        if not self._workspace_allows_square(self.latest_gantry):
            self._stop('square no longer fits inside workspace at motion start')
            return False
        self.origin_x_mm = 1000.0 * float(self.latest_gantry.x)
        self.origin_y_mm = 1000.0 * float(self.latest_gantry.y)
        self.motion_start_wall = now
        self.phase = 'running'
        nzic_suffix = ''
        if self.args.nonzero_ic:
            nzic_suffix = f'; NZIC corrections={len(self.profile.ic_corrections)}'
        self.get_logger().info(
            f'Motion started at ({self.origin_x_mm:.1f}, '
            f'{self.origin_y_mm:.1f}) mm{nzic_suffix}.'
        )
        return True

    def _tick(self) -> None:
        if self.done or self.phase in ('waiting', 'arming'):
            return
        now = time.monotonic()
        if self.phase == 'preflight':
            self._preflight_ready(now)
            return
        if not self._motion_state_is_safe(report=True):
            return
        assert self.motion_start_wall is not None
        motion_time_s = now - self.motion_start_wall

        if self.phase == 'running':
            if (
                self.args.nonzero_ic
                and not self.y_nzic_started
                and motion_time_s >= self.profile.legs[0].shaped_stop_s
            ):
                if not self._nzic_y_ready_to_start(now, motion_time_s):
                    self._publish(0.0, 0.0, motion_time_s)
                    self._periodic_status(motion_time_s, 0.0, 0.0)
                    self._write_log(motion_time_s)
                    return
            if motion_time_s >= self.profile.duration_s:
                self._publish(0.0, 0.0, motion_time_s)
                self.phase = 'settling'
                self.settle_start_wall = now
                self.get_logger().info(
                    'Square complete; command is zero and residual sway is being measured.'
                )
            else:
                vx, vy = self.profile.command_at(motion_time_s)
                if not self._live_workspace_allows(vx, vy):
                    self._stop('live workspace guard blocked the commanded direction')
                    return
                if self._path_error_exceeded(motion_time_s):
                    return
                self._publish(vx, vy, motion_time_s)
                self._periodic_status(motion_time_s, vx, vy)
        elif self.phase == 'settling':
            self._publish(0.0, 0.0, motion_time_s)
            if math.isfinite(self.pitch_deg) and math.isfinite(self.roll_deg):
                self.settle_samples.append((self.pitch_deg, self.roll_deg))
            if now - self.settle_start_wall >= self.args.settle_time_s:
                self._complete()
                return
        self._write_log(motion_time_s)

    def _nzic_y_ready_to_start(self, now: float, motion_time_s: float) -> bool:
        """Plan and validate +Y from live roll data during the first corner."""
        primary_mode = self.profile.modes[0]
        if self.y_hold_start_wall is None:
            self.y_hold_start_wall = now
            self.get_logger().info(
                '+X complete; holding zero while the +Y NZIC correction is '
                'computed from the latest roll state.'
            )
        if now - self.y_hold_start_wall > self.args.ic_y_plan_timeout_s:
            self._stop('no feasible live +Y NZIC phase was found before timeout')
            return False

        y_leg = self.profile.legs[1]
        if (
            not self.y_nzic_ready
            and y_leg.start_s <= motion_time_s + 0.05
        ):
            self.profile = self.profile.with_retimed_leg_start(
                1, motion_time_s + self.args.ic_planning_lead_s
            )
            y_leg = self.profile.legs[1]
        if self.y_nzic_ready:
            if motion_time_s >= y_leg.start_s:
                self.y_nzic_started = True
                self.get_logger().info(
                    f'+Y NZIC plan entered at t={motion_time_s:.3f}s; '
                    f'corrections={len(self.profile.ic_corrections)}.'
                )
                return True
            if now - self.last_y_validation_wall >= self.args.ic_replan_period_s:
                self.last_y_validation_wall = now
                try:
                    _, roll = self._estimate_sway(now, required_axes=('roll',))
                except ValueError as exc:
                    self.y_nzic_ready = False
                    self._periodic_info(f'+Y NZIC estimate rejected: {exc}')
                    return False
                forecast = propagate_sway_state(
                    roll.state, primary_mode, y_leg.start_s - motion_time_s
                )
                assert self.planned_roll_state is not None
                error = SwayState(
                    forecast.angle_rad - self.planned_roll_state.angle_rad,
                    forecast.angular_rate_rad_s
                    - self.planned_roll_state.angular_rate_rad_s,
                ).amplitude_rad(primary_mode)
                if error > math.radians(
                    self.args.ic_plan_validation_tolerance_deg
                ):
                    self.y_nzic_ready = False
                    self._periodic_info(
                        '+Y NZIC state forecast changed; delaying and replanning '
                        f'(error={math.degrees(error):.3f}deg)'
                    )
                    return False
            return False

        if now - self.last_y_plan_attempt_wall < self.args.ic_replan_period_s:
            return False
        self.last_y_plan_attempt_wall = now
        try:
            _, roll = self._estimate_sway(now, required_axes=('roll',))
        except ValueError as exc:
            self._periodic_info(f'+Y NZIC estimator waiting: {exc}')
            return False

        target_start_s = max(
            y_leg.start_s,
            motion_time_s + self.args.ic_planning_lead_s,
        )
        target_roll = propagate_sway_state(
            roll.state, primary_mode, target_start_s - motion_time_s
        )
        try:
            candidate = self.profile.with_retimed_leg_start(1, target_start_s)
            candidate = candidate.with_leg_nonzero_initial_condition(
                leg_index=1,
                initial_state=target_roll,
                initial_state_time_s=target_start_s,
                minimum_correction_amplitude_rad=math.radians(
                    self.args.ic_min_amplitude_deg
                ),
                primary_band_fraction=self.args.ic_primary_band_fraction,
                second_mode_band_fraction=self.args.ic_second_band_fraction,
                maximum_second_mode_residual_fraction=(
                    self.args.ic_max_second_residual_fraction
                ),
            )
        except ValueError as exc:
            self._periodic_info(
                '+Y NZIC phase not yet feasible; extending the zero-velocity '
                f'dwell and retrying: {exc}'
            )
            return False
        finished_motion_time = time.monotonic() - self.motion_start_wall
        if finished_motion_time >= target_start_s - 0.05:
            self._periodic_info(
                '+Y NZIC solve missed its forecast window; delaying and replanning'
            )
            return False

        self.profile = candidate
        self.planned_roll_state = target_roll
        self.y_nzic_ready = True
        self.last_y_validation_wall = now
        y_correction = next(
            (
                correction
                for correction in candidate.ic_corrections
                if correction.leg_name == '+Y'
            ),
            None,
        )
        details = 'standard ZVD (roll below correction threshold)'
        if y_correction is not None:
            details = (
                f'A0={math.degrees(y_correction.initial_amplitude_rad):.3f}deg, '
                f'nominal={math.degrees(y_correction.nominal_residual_rad):.3e}deg, '
                f'rope-band={math.degrees(y_correction.primary_band_residual_rad):.3f}deg, '
                f'high-band={y_correction.second_mode_band_residual_fraction:.4f}'
            )
        self.get_logger().info(
            f'+Y NZIC plan accepted for t={target_start_s:.3f}s; {details}'
        )
        return False

    def _motion_state_is_safe(self, *, report: bool) -> bool:
        if not self._fresh(self.gantry_wall, self.args.state_fresh_timeout_s):
            if report:
                self._stop('/gantry/state became stale')
            return False
        if self.args.require_payload and not self._fresh(
            self.payload_wall, self.args.payload_fresh_timeout_s
        ):
            if report:
                self._stop(f'{self.args.payload_topic} became stale')
            return False
        assert self.latest_gantry is not None
        unsafe = None
        if self.latest_gantry.estop:
            unsafe = 'gantry E-stop became active'
        elif not self.latest_gantry.homed:
            unsafe = 'gantry lost homed state'
        elif not self.latest_gantry.enabled:
            unsafe = 'gantry is not enabled'
        elif self.latest_gantry.mode != 'TRAJ':
            unsafe = f'gantry left TRAJ mode ({self.latest_gantry.mode})'
        if unsafe is not None:
            if report:
                self._stop(unsafe)
            return False
        return True

    def _workspace_allows_square(self, state: GantryState) -> bool:
        low = self.args.workspace_min_mm + self.args.workspace_margin_mm
        high = self.args.workspace_max_mm - self.args.workspace_margin_mm
        x0 = 1000.0 * float(state.x)
        y0 = 1000.0 * float(state.y)
        return (
            low <= x0 <= high
            and low <= y0 <= high
            and x0 + self.profile.x_distance_mm <= high
            and y0 + self.profile.y_distance_mm <= high
        )

    def _live_workspace_allows(self, vx: float, vy: float) -> bool:
        assert self.latest_gantry is not None
        low = self.args.workspace_min_mm + self.args.workspace_margin_mm
        high = self.args.workspace_max_mm - self.args.workspace_margin_mm
        x = 1000.0 * float(self.latest_gantry.x)
        y = 1000.0 * float(self.latest_gantry.y)
        return not (
            (vx < 0.0 and x <= low)
            or (vx > 0.0 and x >= high)
            or (vy < 0.0 and y <= low)
            or (vy > 0.0 and y >= high)
        )

    def _path_error_exceeded(self, motion_time_s: float) -> bool:
        if self.args.max_path_error_mm <= 0.0:
            return False
        assert self.latest_gantry is not None
        assert self.origin_x_mm is not None and self.origin_y_mm is not None
        expected_x, expected_y = self.profile.displacement_at(motion_time_s)
        actual_x = 1000.0 * float(self.latest_gantry.x) - self.origin_x_mm
        actual_y = 1000.0 * float(self.latest_gantry.y) - self.origin_y_mm
        error = math.hypot(actual_x - expected_x, actual_y - expected_y)
        if error > self.args.max_path_error_mm:
            self._stop(
                f'cart path error {error:.1f} mm exceeded '
                f'{self.args.max_path_error_mm:.1f} mm'
            )
            return True
        return False

    def _publish(self, vx_mm_s: float, vy_mm_s: float, motion_time_s: float) -> None:
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = TrajCmd.STREAM
        msg.time_s = float(motion_time_s)
        msg.vx_mm_s = float(vx_mm_s)
        msg.vy_mm_s = float(vy_mm_s)
        if self.latest_gantry is not None:
            msg.x = float(self.latest_gantry.x)
            msg.y = float(self.latest_gantry.y)
        self.traj_pub.publish(msg)

    def _write_log(self, motion_time_s: float) -> None:
        if self.csv_writer is None or self.latest_gantry is None:
            return
        x_mm = 1000.0 * float(self.latest_gantry.x)
        y_mm = 1000.0 * float(self.latest_gantry.y)
        relative_x = float('nan') if self.origin_x_mm is None else x_mm - self.origin_x_mm
        relative_y = float('nan') if self.origin_y_mm is None else y_mm - self.origin_y_mm
        expected_x, expected_y = self.profile.displacement_at(motion_time_s)
        cmd_x, cmd_y = (
            self.profile.command_at(motion_time_s)
            if self.phase == 'running'
            else (0.0, 0.0)
        )
        angle_mag = math.hypot(self.pitch_deg, self.roll_deg)
        planned_pitch = self.planned_pitch_state or SwayState(
            float('nan'), float('nan')
        )
        planned_roll = self.planned_roll_state or SwayState(
            float('nan'), float('nan')
        )
        pitch_fit_rmse = (
            float('nan')
            if self.pitch_estimate is None
            else math.degrees(self.pitch_estimate.rmse_rad)
        )
        roll_fit_rmse = (
            float('nan')
            if self.roll_estimate is None
            else math.degrees(self.roll_estimate.rmse_rad)
        )
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds * 1.0e-9,
            motion_time_s,
            self.phase if self.phase != 'running' else self.profile.phase_at(motion_time_s),
            cmd_x,
            cmd_y,
            x_mm,
            y_mm,
            relative_x,
            relative_y,
            expected_x,
            expected_y,
            relative_x - expected_x,
            relative_y - expected_y,
            1000.0 * float(self.latest_gantry.vx),
            1000.0 * float(self.latest_gantry.vy),
            self.payload_time,
            self.pitch_deg,
            self.roll_deg,
            angle_mag,
            int(self.args.nonzero_ic),
            math.degrees(planned_pitch.angle_rad),
            math.degrees(planned_pitch.angular_rate_rad_s),
            math.degrees(planned_roll.angle_rad),
            math.degrees(planned_roll.angular_rate_rad_s),
            pitch_fit_rmse,
            roll_fit_rmse,
            len(self.profile.ic_corrections),
        ])
        now = time.monotonic()
        if self.csv_file is not None and now - self.last_csv_flush_wall >= 0.5:
            self.csv_file.flush()
            self.last_csv_flush_wall = now

    def _periodic_info(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_status_wall >= self.args.print_period_s:
            self.get_logger().info(message)
            self.last_status_wall = now

    def _periodic_status(self, motion_time_s: float, vx: float, vy: float) -> None:
        now = time.monotonic()
        if now - self.last_status_wall < self.args.print_period_s:
            return
        expected_x, expected_y = self.profile.displacement_at(motion_time_s)
        self.get_logger().info(
            f'{self.profile.phase_at(motion_time_s)} t={motion_time_s:.2f}/'
            f'{self.profile.duration_s:.2f}s cmd=({vx:.1f},{vy:.1f})mm/s '
            f'expected=({expected_x:.1f},{expected_y:.1f})mm '
            f'angle=({self.pitch_deg:.3f},{self.roll_deg:.3f})deg'
        )
        self.last_status_wall = now

    def _complete(self) -> None:
        self._publish(0.0, 0.0, self.profile.duration_s + self.args.settle_time_s)
        residual = 'no payload samples'
        if self.settle_samples:
            magnitudes = [math.hypot(pitch, roll) for pitch, roll in self.settle_samples]
            rms = math.sqrt(sum(value * value for value in magnitudes) / len(magnitudes))
            residual = f'2-D residual peak={max(magnitudes):.4f}deg RMS={rms:.4f}deg'
        self.success = True
        self.done = True
        self.phase = 'complete'
        if self.csv_file is not None:
            self.csv_file.flush()
        suffix = '' if self.log_path is None else f'; log={self.log_path}'
        self.get_logger().info(f'Robust 2-D square complete; {residual}{suffix}')

    def _stop(self, reason: str) -> None:
        if self.done:
            return
        for _ in range(3):
            self._publish(0.0, 0.0, 0.0)
        self.done = True
        self.success = False
        self.phase = 'aborted'
        if self.csv_file is not None:
            self.csv_file.flush()
        self.get_logger().error(f'ABORTED: {reason}; zero velocity commanded')

    def close(self) -> None:
        for _ in range(3):
            self._publish(0.0, 0.0, 0.0)
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='Required to arm and move hardware; otherwise preview only.')
    parser.add_argument('--x-distance-mm', type=float, default=1000.0)
    parser.add_argument('--y-distance-mm', type=float, default=1000.0)
    parser.add_argument('--speed-mm-s', type=float, default=400.0)
    parser.add_argument('--rope-length-m', type=float, default=0.90)
    parser.add_argument('--damping-ratio', type=float, default=0.0)
    parser.add_argument('--timing-scale', type=float, default=1.0)
    parser.add_argument('--second-mode-hz', type=float, default=5.06,
                        help='Second natural mode; use 0 to disable (default: 5.06).')
    parser.add_argument('--second-mode-damping-ratio', type=float, default=0.0)
    parser.add_argument('--second-mode-timing-scale', type=float, default=1.0)
    parser.add_argument('--corner-dwell-s', type=float, default=1.0)
    parser.add_argument('--settle-time-s', type=float, default=3.0)
    parser.add_argument('--start-delay-s', type=float, default=1.0)
    parser.add_argument('--start-hold-s', type=float, default=0.50,
                        help='Time the cart must remain inside its speed limit.')
    parser.add_argument('--start-cart-speed-limit-mm-s', type=float, default=5.0)
    parser.add_argument('--start-angle-limit-deg', type=float, default=1.0,
                        help='Payload rest limit used only with --zero-ic-only.')
    parser.add_argument(
        '--zero-ic-only', dest='nonzero_ic', action='store_false',
        help='Disable live nonzero-initial-condition estimation and require rest.',
    )
    parser.set_defaults(nonzero_ic=True)
    parser.add_argument('--ic-estimator-window-s', type=float, default=2.5)
    parser.add_argument('--ic-estimator-min-samples', type=int, default=120)
    parser.add_argument('--ic-estimator-min-span-s', type=float, default=1.5)
    parser.add_argument('--ic-max-fit-rmse-deg', type=float, default=0.10)
    parser.add_argument(
        '--ic-min-amplitude-deg', type=float, default=0.10,
        help='Below this fitted sway amplitude, retain the standard ZVD weights.',
    )
    parser.add_argument(
        '--ic-max-amplitude-deg', type=float, default=1.50,
        help='Refuse to start above this fitted per-axis sway amplitude.',
    )
    parser.add_argument('--ic-primary-band-fraction', type=float, default=0.10)
    parser.add_argument(
        '--ic-second-band-fraction', type=float, default=0.025,
        help='Robust band around the measured higher mode (default +/-2.5%%).',
    )
    parser.add_argument(
        '--ic-max-second-residual-fraction', type=float, default=0.03,
    )
    parser.add_argument('--ic-planning-lead-s', type=float, default=0.50)
    parser.add_argument(
        '--ic-plan-validation-tolerance-deg', type=float, default=0.12,
    )
    parser.add_argument('--ic-replan-period-s', type=float, default=0.10)
    parser.add_argument('--ic-y-plan-timeout-s', type=float, default=12.0)
    parser.add_argument('--rate-hz', type=float, default=100.0)
    parser.add_argument('--payload-topic', default='/payload/pose_e_rel')
    parser.add_argument('--allow-missing-payload', dest='require_payload',
                        action='store_false',
                        help='Do not require fresh payload angles (not recommended).')
    parser.set_defaults(require_payload=True)
    parser.add_argument('--payload-fresh-timeout-s', type=float, default=0.30)
    parser.add_argument('--state-fresh-timeout-s', type=float, default=0.30)
    parser.add_argument('--arm-timeout-s', type=float, default=20.0)
    parser.add_argument('--service-wait-timeout-s', type=float, default=15.0,
                        help='DDS service-discovery grace period before aborting.')
    parser.add_argument('--workspace-min-mm', type=float, default=0.0)
    parser.add_argument('--workspace-max-mm', type=float, default=1150.0)
    parser.add_argument('--workspace-margin-mm', type=float, default=5.0)
    parser.add_argument('--max-path-error-mm', type=float, default=75.0,
                        help='Abort on 2-D measured/predicted path error; <=0 disables.')
    parser.add_argument('--print-period-s', type=float, default=0.5)
    parser.add_argument('--log-csv', default='auto',
                        help="CSV path, 'auto' for ./logs/robust_2d, or empty to disable.")
    return parser


def _make_profile(args: argparse.Namespace) -> Robust2dSquareProfile:
    return Robust2dSquareProfile(
        x_distance_mm=args.x_distance_mm,
        y_distance_mm=args.y_distance_mm,
        speed_mm_s=args.speed_mm_s,
        rope_length_m=args.rope_length_m,
        damping_ratio=args.damping_ratio,
        timing_scale=args.timing_scale,
        second_mode_frequency_hz=(
            None if args.second_mode_hz == 0.0 else args.second_mode_hz
        ),
        second_mode_damping_ratio=args.second_mode_damping_ratio,
        second_mode_timing_scale=args.second_mode_timing_scale,
        corner_dwell_s=args.corner_dwell_s,
    )


def _validate_runtime_args(args: argparse.Namespace) -> None:
    positive = (
        args.settle_time_s,
        args.rate_hz,
        args.payload_fresh_timeout_s,
        args.state_fresh_timeout_s,
        args.arm_timeout_s,
        args.service_wait_timeout_s,
        args.print_period_s,
        args.start_cart_speed_limit_mm_s,
        args.start_angle_limit_deg,
        args.ic_estimator_window_s,
        args.ic_estimator_min_span_s,
        args.ic_max_fit_rmse_deg,
        args.ic_max_amplitude_deg,
        args.ic_max_second_residual_fraction,
        args.ic_planning_lead_s,
        args.ic_plan_validation_tolerance_deg,
        args.ic_replan_period_s,
        args.ic_y_plan_timeout_s,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError('settle/rate/timeout/print arguments must be finite and positive')
    if not math.isfinite(args.start_delay_s) or args.start_delay_s < 0.0:
        raise ValueError('start-delay-s must be finite and nonnegative')
    if not math.isfinite(args.start_hold_s) or args.start_hold_s < 0.0:
        raise ValueError('start-hold-s must be finite and nonnegative')
    if not math.isfinite(args.workspace_margin_mm) or args.workspace_margin_mm < 0.0:
        raise ValueError('workspace-margin-mm must be finite and nonnegative')
    if args.workspace_max_mm <= args.workspace_min_mm + 2.0 * args.workspace_margin_mm:
        raise ValueError('workspace bounds and margin leave no usable interval')
    if args.ic_estimator_min_samples < 3:
        raise ValueError('ic-estimator-min-samples must be at least 3')
    if args.ic_estimator_min_span_s > args.ic_estimator_window_s:
        raise ValueError('IC estimator minimum span cannot exceed its window')
    if not math.isfinite(args.ic_min_amplitude_deg) or args.ic_min_amplitude_deg < 0.0:
        raise ValueError('ic-min-amplitude-deg must be finite and nonnegative')
    if args.ic_min_amplitude_deg >= args.ic_max_amplitude_deg:
        raise ValueError('IC minimum amplitude must be below its maximum')
    if not 0.0 < args.ic_primary_band_fraction < 0.5:
        raise ValueError('ic-primary-band-fraction must be in (0, 0.5)')
    if not 0.0 < args.ic_second_band_fraction < 0.5:
        raise ValueError('ic-second-band-fraction must be in (0, 0.5)')
    if args.nonzero_ic and not args.require_payload:
        raise ValueError('--allow-missing-payload requires --zero-ic-only')


def print_preview(
    profile: Robust2dSquareProfile,
    *,
    execute: bool = False,
    nonzero_ic: bool = True,
) -> None:
    heading = (
        'Robust 2-D multi-mode ZVD square execution configuration'
        if execute
        else 'Robust 2-D multi-mode ZVD square preview (no hardware motion)'
    )
    print(heading)
    for mode in profile.modes:
        print(
            f'  {mode.name}: f_n={mode.natural_frequency_hz:.4f}Hz, '
            f'zeta={mode.damping_ratio:.4f}, '
            f'T={mode.impulse_spacing_s:.4f}s, A={mode.amplitudes}'
        )
    print(
        f'  convolved shaper: {len(profile.amplitudes)} impulses, '
        f'tail={profile.shaper_tail_s:.4f}s, '
        f'minimum monotonic leg={profile.minimum_monotonic_leg_distance_mm:.1f}mm'
    )
    print('  impulses:')
    for index, (time_s, amplitude) in enumerate(
        zip(profile.impulse_times_s, profile.amplitudes), start=1
    ):
        print(f'    {index:>2}: t={time_s:7.4f}s A={amplitude:.8f}')
    print(
        f'  path: +X {profile.x_distance_mm:.1f}mm, '
        f'+Y {profile.y_distance_mm:.1f}mm, -X, -Y; '
        f'total shaped motion={profile.duration_s:.3f}s'
    )
    for name, start, raw_stop, shaped_stop, distance in profile.iter_summary_rows():
        print(
            f'  {name:>2}: start={start:7.3f}s raw_stop={raw_stop:7.3f}s '
            f'shaped_stop={shaped_stop:7.3f}s signed_distance={distance:8.1f}mm'
        )
    print(
        '  rope-mode residual: nominal='
        f'{profile.residual_fraction():.3e}, '
        f'-10% freq={profile.residual_fraction(0.90):.4f}, '
        f'+10% freq={profile.residual_fraction(1.10):.4f}'
    )
    if len(profile.modes) > 1:
        print(
            '  second-mode residual: nominal='
            f'{profile.second_mode_residual_fraction():.3e}, '
            f'-10% freq={profile.second_mode_residual_fraction(0.90):.4f}, '
            f'+10% freq={profile.second_mode_residual_fraction(1.10):.4f}'
        )
    if nonzero_ic:
        print(
            '  initial condition: live encoder harmonic regression + monotonic '
            'robust NZIC replanning enabled at execution time'
        )
    else:
        print('  initial condition: measured-rest gate (NZIC disabled)')
    if not execute:
        print('Add --execute to arm and run this profile.')


def main() -> int:
    args = build_parser().parse_args()
    try:
        _validate_runtime_args(args)
        profile = _make_profile(args)
    except ValueError as exc:
        print(f'Invalid configuration: {exc}', file=sys.stderr)
        return 2
    print_preview(profile, execute=args.execute, nonzero_ic=args.nonzero_ic)
    if not args.execute:
        return 0

    rclpy.init()
    node = Robust2dSquarePlayer(args, profile)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node._stop('operator interrupted with Ctrl+C')
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if node.success else 1


if __name__ == '__main__':
    raise SystemExit(main())
