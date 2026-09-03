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

The "robust" profile is a model-based ZVD baseline. The "zv" profile is the
corresponding non-robust two-impulse ZV baseline. Both compute T and weights
from a configured rope length and do not use online ID.

The "nonzero-ic" profile observes a stationary-trolley free swing for ``tau``
seconds, fits its frequency and instantaneous state, then uses the undamped
closed-form two-impulse solution to cancel that measured initial state at the
end of the move.  It holds zero until the fitted state permits a forward-only
command.

With ``--excite`` the "nonzero-ic" profile is preceded by a closed-loop
swing-up stage (``nonzero_ic_exciter.NonzeroIcExciter``).  A resonant,
amplitude-regulating trolley drive pumps the payload up to
``--excite-target-angle-deg`` of axis swing (encoder angle), converging without
overshoot and handing off at a swing extremum.  The ``tau`` free-swing
identification window and the rest of the profile then run unchanged, timed
from swing-up completion.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
import sympy as sy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from payload_perception_msgs.msg import PayloadState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import ExecuteTimedProfile, SetMode

from adaptive_id_player import AdaptiveIdentifier, Estimate
from nonzero_ic_exciter import BoundedCycleExciter, BoundedExciteConfig
from nonzero_ic_shaper import (
    FreeSwingFrequencyEstimate,
    RobustNonzeroIcShaper,
    correct_finite_amplitude_frequency,
    estimate_free_swing_frequency,
    solve_nonzero_ic_shaper,
    solve_robust_nonzero_ic_shaper,
)


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


@dataclass
class GantryStateSample:
    rx_wall: float
    stamp: float
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True)
class TimedPayloadObservation:
    """Payload sample nearest a controller-timed event in payload-clock time."""

    sample_time_s: float
    target_time_s: float
    swing_mm: float
    world_payload_velocity_mm_s: float

    @property
    def offset_ms(self) -> float:
        return 1000.0 * (self.sample_time_s - self.target_time_s)


def closer_timed_payload_observation(
    current: TimedPayloadObservation | None,
    *,
    sample_time_s: float,
    target_time_s: float,
    swing_mm: float,
    world_payload_velocity_mm_s: float,
) -> TimedPayloadObservation:
    """Keep the measurement closest to an event using the sensor time base."""
    candidate = TimedPayloadObservation(
        sample_time_s=float(sample_time_s),
        target_time_s=float(target_time_s),
        swing_mm=float(swing_mm),
        world_payload_velocity_mm_s=float(world_payload_velocity_mm_s),
    )
    if current is None or abs(candidate.offset_ms) < abs(current.offset_ms):
        return candidate
    return current


class NonzeroIcNotReady(ValueError):
    """The live initial state has no forward-only two-impulse solution yet."""


def scaled_nonzero_ic_frequency(
    fitted_omega_rad_s: float,
    scale: float,
) -> float:
    """Return the experimental shaper frequency without changing the ID fit."""
    return float(fitted_omega_rad_s) * float(scale)


def select_nonzero_ic_shaper_frequency(
    fitted_omega_rad_s: float,
    scale: float,
    fixed_omega_rad_s: float,
) -> float:
    """Select an absolute experimental omega or a scaled fitted omega."""
    if float(fixed_omega_rad_s) > 0.0:
        return float(fixed_omega_rad_s)
    return scaled_nonzero_ic_frequency(fitted_omega_rad_s, scale)


def actuator_compensated_profile_start(
    fitted_peak_wall: float,
    actuation_lead_ms: float,
) -> float:
    """Issue the controller profile ahead of the fitted physical swing peak."""
    return float(fitted_peak_wall) - 1.0e-3 * float(actuation_lead_ms)


def update_payload_clock_calibration(
    calibrated_offset_s: float | None,
    previous_payload_time_s: float | None,
    payload_time_s: float,
    receive_wall_s: float,
) -> tuple[float, float, bool]:
    """Update the publisher-time to local-monotonic clock calibration.

    ``receive_wall_s - payload_time_s`` contains the fixed clock offset plus
    nonnegative ROS/executor queueing delay.  The minimum observed offset is
    therefore the best online estimate of the clock mapping.  A large source
    timestamp regression denotes an encoder-node clock reset and starts a new
    calibration epoch.

    Returns ``(clock_offset_s, queue_delay_s, clock_reset)``.
    """
    payload_time_s = float(payload_time_s)
    receive_wall_s = float(receive_wall_s)
    if not math.isfinite(payload_time_s) or not math.isfinite(receive_wall_s):
        raise ValueError('payload and receive times must be finite')

    clock_reset = (
        previous_payload_time_s is not None
        and math.isfinite(float(previous_payload_time_s))
        and payload_time_s < float(previous_payload_time_s) - 1.0e-6
    )
    observed_offset_s = receive_wall_s - payload_time_s
    if (
        calibrated_offset_s is None
        or not math.isfinite(float(calibrated_offset_s))
        or clock_reset
    ):
        calibrated_offset_s = observed_offset_s
    else:
        calibrated_offset_s = min(
            float(calibrated_offset_s), observed_offset_s)
    queue_delay_s = max(0.0, observed_offset_s - calibrated_offset_s)
    return float(calibrated_offset_s), float(queue_delay_s), clock_reset


def payload_time_at_wall(
    wall_time_s: float,
    calibrated_offset_s: float,
) -> float:
    """Convert local monotonic time to the payload publisher's time base."""
    return float(wall_time_s) - float(calibrated_offset_s)


def payload_wall_at_time(
    payload_time_s: float,
    calibrated_offset_s: float,
) -> float:
    """Convert payload publisher time to calibrated local monotonic time."""
    return float(payload_time_s) + float(calibrated_offset_s)


class AdaptivePaperTdfPlayer(Node):
    def __init__(self, args):
        super().__init__('adaptive_paper_tdf_player')
        self.args = args

        self.latest_gantry_state: GantryState | None = None
        self.latest_gantry_state_sample: GantryStateSample | None = None
        self.latest_cart_state_sample: GantryStateSample | None = None
        self.gantry_state_samples: list[GantryStateSample] = []
        self.latest_payload_abs_mm: float | None = None
        self.latest_swing_mm: float | None = None
        self.latest_payload_rel_x_mm: float | None = None
        self.latest_payload_rel_y_mm: float | None = None
        self.latest_payload_rel_z_mm: float | None = None
        self.latest_payload_rel_vx_mm_s: float | None = None
        self.latest_payload_rel_vy_mm_s: float | None = None
        self.latest_payload_rel_raw_vx_mm_s: float | None = None
        self.latest_payload_rel_raw_vy_mm_s: float | None = None
        self.latest_payload_encoder_sample_age_ms: float | None = None
        self.latest_cart_q_mm: float | None = None
        self.latest_cam_gantry_x_mm: float | None = None
        self.latest_cam_gantry_y_mm: float | None = None
        self.latest_payload_time: float | None = None
        self.latest_payload_wall: float | None = None
        self.payload_clock_offset_s: float | None = None
        self.latest_payload_observed_clock_offset_s: float | None = None
        self.latest_payload_queue_delay_ms: float | None = None
        self.latest_traj_latency: list[float] | None = None
        self.latest_traj_latency_wall: float | None = None
        self.latest_id_filtered_swing_mm: float | None = None
        self._id_filter_time: float | None = None

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
        self.mode_fit_samples: list[tuple[float, float]] = []
        self.zero_zeta_samples: list[tuple[float, float]] = []
        self.latest_zero_zeta_estimate: Estimate | None = None
        self.latest_zero_zeta_score: float | None = None
        self.latest_zero_zeta_rmse: float | None = None
        self.latest_zero_zeta_amp: float | None = None
        self.latest_freq_bank_estimate: Estimate | None = None
        self.latest_paper2_estimate: Estimate | None = None
        self.latest_two_mode_estimate: Estimate | None = None
        self.latest_two_mode_t2: float | None = None
        self.latest_two_mode_amp1: float | None = None
        self.latest_two_mode_amp2: float | None = None
        self.latest_two_mode_amp_ratio: float | None = None
        self.latest_two_mode_nrmse: float | None = None
        self._last_mode_fit_log_t = -1.0
        self._last_zero_zeta_update_t = -1.0
        self._zero_zeta_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix='zero-zeta-id')
            if args.id_method == 'zero-zeta-ls'
            else None
        )
        self._zero_zeta_future: Future | None = None
        self._last_freq_bank_update_t = -1.0
        self._last_two_mode_update_t = -1.0
        self.residual_samples: list[tuple[float, float]] = []
        self.csv_file = None
        self.csv_writer: csv.writer | None = None
        self.csv_path: Path | None = None
        self.run_outputs_generated = False
        self._log_flush_count = 0
        self._last_wait_ready_print = 0.0
        self._last_id_status_print = 0.0
        self._auto_start_ready_wall: float | None = None
        self._last_nonzero_ic_wait_log_wall = -math.inf
        self._last_nonzero_ic_wait_reason = ''
        self._id_updates_stopped_logged = False
        self._missed_first_switch_warned = False
        self._last_stream_wall: float | None = None
        self.latest_stream_dt_ms: float | None = None
        self.latest_payload_age_ms: float | None = None
        self.latest_gantry_state_age_ms: float | None = None
        self.nonzero_ic_initial_swing_mm: float | None = None
        self.nonzero_ic_initial_payload_velocity_mm_s: float | None = None
        self.nonzero_ic_predicted_peak_swing_mm: float | None = None
        self.nonzero_ic_predicted_peak_payload_velocity_mm_s: float | None = None
        self.nonzero_ic_predicted_command_start_swing_mm: float | None = None
        self.nonzero_ic_predicted_command_start_payload_velocity_mm_s: float | None = None
        self.nonzero_ic_fitted_omega_rad_s: float | None = None
        self.nonzero_ic_small_angle_omega_rad_s: float | None = None
        self.nonzero_ic_fitted_amplitude_angle_deg: float | None = None
        self.nonzero_ic_frequency_correction_factor: float | None = None
        self.nonzero_ic_used_omega_rad_s: float | None = None
        self.nonzero_ic_robust_solution: RobustNonzeroIcShaper | None = None
        self.nonzero_ic_profile_events: tuple[tuple[float, float], ...] | None = None
        self.nonzero_ic_swing_samples: list[tuple[float, float]] = []
        self.nonzero_ic_frequency_estimate: FreeSwingFrequencyEstimate | None = None
        self._nonzero_ic_fit_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix='nonzero-ic-id')
            if args.profile == 'nonzero-ic' and args.nonzero_ic_adaptive_frequency
            else None
        )
        self._nonzero_ic_fit_future: Future | None = None
        self._last_nonzero_ic_fit_wall = -math.inf
        self._last_nonzero_ic_fit_log_wall = -math.inf
        self._last_nonzero_ic_fit_error = 'frequency fit has not run yet'
        self.timed_profile_future: Future | None = None
        self.timed_profile_start_wall: float | None = None
        self.timed_profile_peak_wall: float | None = None
        self.timed_profile_start_payload_time: float | None = None
        self.timed_profile_peak_payload_time: float | None = None
        self.timed_profile_payload_clock_offset_s: float | None = None
        self.timed_profile_arm_payload_queue_delay_ms: float | None = None
        self.nonzero_ic_measured_command_start: TimedPayloadObservation | None = None
        self.nonzero_ic_measured_peak: TimedPayloadObservation | None = None
        self.timed_profile_request_wall: float | None = None
        self.controller_timed_profile_active = False

        # Closed-loop swing-up (``--excite``) state.  ``phase`` tracks the
        # nonzero-IC state machine: wait_telemetry -> excite -> id_hold ->
        # wait_peak -> arm_profile -> armed_profile -> maneuver -> residual.
        # Profiles without ``--excite`` and non-nonzero-ic profiles skip the
        # excitation/ID phases as appropriate.
        self.phase = 'wait_telemetry'
        self.run_start_wall: float | None = None
        self.id_hold_start_wall: float | None = None
        self.excite_start_cart_q_mm: float | None = None
        self.excite_angle_bias_deg = 0.0
        self.latest_enc_wall: float | None = None
        self.latest_enc_diag_wall: float | None = None
        self._latest_excite_cmd = None
        self.excite_only_complete = False
        self._last_excite_wait_log_wall = -math.inf
        self.exciter: BoundedCycleExciter | None = None
        if args.profile == 'nonzero-ic' and args.excite:
            excite_omega = (
                args.excite_omega_rad_s
                if args.excite_omega_rad_s > 0.0
                else math.sqrt(args.gravity_m_s2 / args.robust_rope_length_m)
            )
            self.exciter = BoundedCycleExciter(BoundedExciteConfig(
                target_angle_deg=args.excite_target_angle_deg,
                omega_rad_s=excite_omega,
                tolerance_deg=args.excite_angle_tolerance_deg,
                speed_mm_s=args.excite_speed_mm_s,
                initial_excursion_mm=args.excite_initial_excursion_mm,
                excursion_step_mm=args.excite_excursion_step_mm,
                max_excursion_mm=args.excite_travel_budget_mm,
                position_kp_s=args.excite_position_kp_s,
                return_speed_mm_s=args.excite_return_speed_mm_s,
                return_tolerance_mm=args.excite_return_tolerance_mm,
                settle_cycles=args.excite_settle_cycles,
                timeout_s=args.excite_timeout_s,
                abort_angle_deg=args.excite_abort_angle_deg,
                slew_mm_s2=args.excite_slew_mm_s2,
            ))

        # Encoder debug logging. These come from:
        #   /payload/pose_e:
        #     [time, pitch_deg, roll_deg, pitch_count, roll_count]
        #   /payload/encoder/diagnostics:
        #     [time, arduino_ms, pitch_raw, roll_raw, pitch_count, roll_count,
        #      imu1_ax, imu1_ay, imu1_az, imu1_gx, imu1_gy, imu1_gz,
        #      imu2_ax, imu2_ay, imu2_az, imu2_gx, imu2_gy, imu2_gz,
        #      optional packet_age_ms, packet_seen, serial_lines, parse_errors, stale]
        self.latest_enc_pitch_deg = None
        self.latest_enc_roll_deg = None
        self.latest_enc_pitch_count = None
        self.latest_enc_roll_count = None
        self.latest_enc_time = None
        self.latest_enc_arduino_ms = None
        self.latest_enc_pitch_raw = None
        self.latest_enc_roll_raw = None
        self.latest_enc_imu1_ax = None
        self.latest_enc_imu1_ay = None
        self.latest_enc_imu1_az = None
        self.latest_enc_imu1_gx = None
        self.latest_enc_imu1_gy = None
        self.latest_enc_imu1_gz = None
        self.latest_enc_imu2_ax = None
        self.latest_enc_imu2_ay = None
        self.latest_enc_imu2_az = None
        self.latest_enc_imu2_gx = None
        self.latest_enc_imu2_gy = None
        self.latest_enc_imu2_gz = None
        self.latest_enc_packet_age_ms = None
        self.latest_enc_packet_seen = None
        self.latest_enc_serial_lines = None
        self.latest_enc_parse_errors = None
        self.latest_enc_stale = None

        self.direction = 1.0 if args.vmax_mm_s >= 0.0 else -1.0
        self.vmax_abs_mm_s = abs(float(args.vmax_mm_s))
        self.tf = args.target_distance_mm / self.vmax_abs_mm_s
        self.initial_velocity_mm_s = self.direction * args.a0 * self.vmax_abs_mm_s
        if args.profile in ('robust', 'zv'):
            self._configure_model_schedule()

        self.identifier = AdaptiveIdentifier(
            K_mm_s=self.initial_velocity_mm_s,
            A0=abs(self.initial_velocity_mm_s) / self.vmax_abs_mm_s,
            cond_threshold=args.cond_threshold,
            omega_min=args.omega_min,
            omega_max=args.omega_max,
            zeta_min=args.id_zeta_min,
            zeta_max=args.id_zeta_max,
            lowpass_hz=0.0,
            local_window_s=args.integral_id_window_s,
            assume_zero_zeta=args.id_assume_zero_zeta,
        )

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.traj_pub = self.create_publisher(TrajCmd, '/traj_cmd', qos)
        self.estimate_pub = self.create_publisher(Float64MultiArray, args.estimate_topic, 10)
        self.enable_cli = self.create_client(Trigger, '/gantry/enable')
        self.mode_cli = self.create_client(SetMode, '/gantry/set_mode')
        self.timed_profile_cli = self.create_client(
            ExecuteTimedProfile, '/gantry/execute_timed_profile')

        self.create_subscription(GantryState, '/gantry/state', self._gantry_state_cb, 10)
        self.create_subscription(TrajCmd, '/traj_cmd', self._traj_cmd_cb, qos)
        self.create_subscription(
            Float64MultiArray, args.payload_topic, self._payload_cb, qos_profile_sensor_data)
        self.create_subscription(
            Float64MultiArray, '/payload/pose_e', self._encoder_pose_cb, qos_profile_sensor_data)
        self.create_subscription(
            Float64MultiArray, '/payload/encoder/diagnostics',
            self._encoder_diag_cb, qos_profile_sensor_data)
        self.create_subscription(
            Float64MultiArray, '/gantry/traj_latency',
            self._traj_latency_cb, qos_profile_sensor_data)
        self.create_subscription(
            PayloadState, '/payload/state',
            self._payload_state_cb, qos_profile_sensor_data)

        if args.log_csv:
            self._open_csv_log(args.log_csv)
        if not args.no_arm:
            self._arm_traj_mode()

        self.stream_timer = self.create_timer(1.0 / args.stream_rate_hz, self._stream_timer_cb)
        self.start_timer = self.create_timer(0.5, self._start_timer_cb)

        self.get_logger().info('adaptive_paper_tdf_player started')
        initial_command = (
            'deferred until MOTION_START'
            if args.profile == 'nonzero-ic'
            else f'{self.initial_velocity_mm_s:.1f} mm/s'
        )
        amplitude_summary = (
            'A0/A1=deferred'
            if args.profile == 'nonzero-ic'
            else f'A0={args.a0:.3f}'
        )
        self.get_logger().info(
            f'Profile={args.profile} axis={args.axis} target={args.target_distance_mm:.1f} mm '
            f'vmax={args.vmax_mm_s:.1f} mm/s {amplitude_summary} '
            f'initial_command={initial_command} tf={self.tf:.3f}s')
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
            self.get_logger().info(f'Online ID method: {args.id_method}')
            if args.id_method == 'integral' and args.integral_id_window_s > 0.0:
                self.get_logger().info(
                    f'Integral ID local window enabled: {args.integral_id_window_s:.3f}s '
                    'with fitted local initial velocity')
            if args.id_assume_zero_zeta:
                self.get_logger().info('Integral ID zero-zeta assumption enabled')
            if args.id_lowpass_hz > 0.0:
                self.get_logger().info(
                    f'ID swing low-pass enabled: cutoff={args.id_lowpass_hz:.2f} Hz')
            else:
                self.get_logger().info('ID input low-pass disabled')
            if args.fixed_id_t_sec > 0.0:
                self.get_logger().info(
                    f'Fixed-ID baseline enabled: T={args.fixed_id_t_sec:.4f}s '
                    f'zeta={args.fixed_id_zeta:.4f}')
            elif args.fixed_id_rope_length_m > 0.0:
                fixed_omega = math.sqrt(args.gravity_m_s2 / args.fixed_id_rope_length_m)
                fixed_T = math.pi / fixed_omega
                self.get_logger().info(
                    f'Fixed-ID baseline enabled: L={args.fixed_id_rope_length_m:.3f}m '
                    f'omega_n={fixed_omega:.4f}rad/s T={fixed_T:.4f}s '
                    f'zeta={args.fixed_id_zeta:.4f}')
            if args.profile == 'colleague-paper-closed':
                self.get_logger().info(
                    f'IS2 selection mode={args.is2_selection_mode} '
                    f'recent_window={args.is2_selection_window_s:.3f}s '
                    f'stable_window={args.is2_stable_window_s:.3f}s '
                    f'stable_after={args.is2_stable_after_s:.3f}s '
                    f'stable_max_range={args.is2_stable_max_range_s:.3f}s')
                if args.is2_id_t_min > 0.0 or args.is2_id_t_max > 0.0:
                    tmax_text = 'inf' if args.is2_id_t_max <= 0.0 else f'{args.is2_id_t_max:.3f}'
                    self.get_logger().info(
                        f'IS2 ID_T gate: [{args.is2_id_t_min:.3f}, {tmax_text}]s')
                self.get_logger().info(
                    f'IS2 schedule feasibility filter: '
                    f'{"enabled" if args.is2_schedule_filter else "disabled"} '
                    f'margin={args.is2_schedule_margin_s:.3f}s')
            if args.profile in (
                'colleague-moving',
                'colleague-velocity-exact',
                'colleague-paper-closed',
            ):
                self.get_logger().info(
                    f'{args.profile} IS: tau={args.tau:.3f}s, initial speed K*vmax, '
                    'then switches at tau+T, tau+2T, tf, tf+tau+T, tf+tau+2T')
        elif args.profile in ('robust', 'zv'):
            sched = self.schedule
            if sched is not None:
                omega = math.sqrt(args.gravity_m_s2 / args.robust_rope_length_m)
                label = 'Robust ZVD' if args.profile == 'robust' else 'Non-robust ZV'
                self.get_logger().info(
                    f'{label} baseline: L={args.robust_rope_length_m:.3f}m '
                    f'omega_n={omega:.4f}rad/s T={sched.T:.4f}s '
                    f'A=[{sched.A0:.4f},{sched.A1:.4f},{sched.A2:.4f}]')
        elif args.profile == 'nonzero-ic':
            if args.nonzero_ic_adaptive_frequency:
                frequency_source = (
                    'adaptive free-swing fit over '
                    f'tau={args.tau:.1f}s in '
                    f'[{args.nonzero_ic_omega_min_rad_s:.3f}, '
                    f'{args.nonzero_ic_omega_max_rad_s:.3f}]rad/s'
                )
            else:
                omega = (
                    args.nonzero_ic_omega_rad_s
                    if args.nonzero_ic_omega_rad_s > 0.0
                    else math.sqrt(
                        args.gravity_m_s2 / args.robust_rope_length_m
                    )
                )
                frequency_source = f'fixed omega_n={omega:.4f}rad/s'
            state_source = (
                'CLI overrides'
                if (
                    args.initial_swing_mm is not None
                    and args.initial_payload_velocity_mm_s is not None
                )
                else 'live payload/cart measurements (with any supplied overrides)'
            )
            self.get_logger().info(
                f'Nonzero-IC {"robust specified-insensitivity" if args.nonzero_ic_robust else "two-impulse"} '
                f'profile: {frequency_source}; '
                f'initial state source={state_source}. Schedule is solved and '
                'terminal-verified immediately before the first command. TRAJ '
                f'activation waits tau={args.tau:.1f}s after '
                'telemetry is ready, then holds zero until both gains are '
                'forward-only.'
                + (
                    ' Start is phase-locked to the positive swing peak in the '
                    'direction of travel with '
                    f'|payload_v| <= '
                    f'{args.nonzero_ic_peak_velocity_tolerance_mm_s:.1f}mm/s.'
                    if args.nonzero_ic_start_at_peak
                    else ''
                ))
            if args.nonzero_ic_robust:
                self.get_logger().info(
                    'Robust nonzero-IC optimization enabled: exact nominal '
                    'terminal cancellation, positive acceleration/deceleration '
                    'increments, exact travel, and minimized residual over '
                    f'+/-{100.0 * args.nonzero_ic_robust_band_fraction:.1f}% '
                    'frequency uncertainty.')
            if args.controller_timed_profile:
                self.get_logger().info(
                    'Switch timing: controller-owned atomic profile; next '
                    'travel-direction peak is predicted with at least '
                    f'{args.timed_profile_lead_s:.3f}s service lead; commands '
                    f'are advanced {args.timed_profile_actuation_lead_ms:.1f}ms '
                    'relative to the fitted physical peak using the '
                    'equivalent-pure-delay actuator model. Predicted and '
                    'nearest measured command/peak states are logged '
                    'separately. Payload-clock mapping uses the minimum '
                    'observed publisher-to-callback offset and arming waits '
                    'for queue delay <= '
                    f'{args.nonzero_ic_max_payload_queue_delay_ms:.1f}ms.')
            if not math.isclose(
                args.nonzero_ic_shaper_frequency_scale,
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                self.get_logger().info(
                    'Experimental nonzero-IC shaper frequency scale: '
                    f'{args.nonzero_ic_shaper_frequency_scale:.6f}; fitted '
                    'frequency remains unchanged for phase/peak prediction.')
            if args.nonzero_ic_shaper_omega_rad_s > 0.0:
                self.get_logger().info(
                    'Experimental fixed nonzero-IC shaper frequency: '
                    f'{args.nonzero_ic_shaper_omega_rad_s:.6f}rad/s; the '
                    'adaptive fit remains active only for state and future-peak '
                    'prediction.')
            if args.nonzero_ic_finite_amplitude_correction:
                self.get_logger().info(
                    'Finite-amplitude frequency correction enabled: the '
                    'sinusoidal fit remains the phase/peak model, while the '
                    'closed-form solver uses the exact simple-pendulum '
                    'small-angle frequency inferred with '
                    f'L={args.robust_rope_length_m:.4f}m.')
            if args.excite and self.exciter is not None:
                self.get_logger().info(
                    'Excite stage enabled: closed-loop swing-up to '
                    f'{args.excite_target_angle_deg:.1f}deg axis swing '
                    f'(omega_n={self.exciter.cfg.omega_rad_s:.4f}rad/s, '
                    f'speed={args.excite_speed_mm_s:.0f}mm/s, '
                    f'tol={args.excite_angle_tolerance_deg:.1f}deg). The '
                    f'tau={args.tau:.1f}s free-swing ID window starts when '
                    'swing-up converges.')
        else:
            self.get_logger().info(
                'Pulse baseline: '
                f'hold zero for {args.pulse_pre_delay_s:.3f}s, command vmax '
                f'for tf={self.tf:.3f}s, then zero.')

    def destroy_node(self):
        try:
            if rclpy.ok():
                if (
                    self.controller_timed_profile_active
                    or self.timed_profile_start_wall is not None
                ):
                    self._publish_traj_abort()
                else:
                    self._publish_stream(0.0, 0.0)
        except Exception:
            pass
        if self.csv_file is not None:
            try:
                self.csv_file.flush()
                self._generate_run_outputs()
            except Exception as exc:
                self.get_logger().warn(
                    f'Automatic run output generation failed during shutdown: {exc}')
            self.csv_file.close()
            self.csv_file = None
        if self._zero_zeta_executor is not None:
            self._zero_zeta_executor.shutdown(wait=True, cancel_futures=True)
            self._zero_zeta_executor = None
        if self._nonzero_ic_fit_executor is not None:
            self._nonzero_ic_fit_executor.shutdown(wait=True, cancel_futures=True)
            self._nonzero_ic_fit_executor = None
        super().destroy_node()

    def _open_csv_log(self, path_arg: str):
        path = Path(path_arg).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = path.resolve()
        self.csv_file = path.open('w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'wall_time_sec',
            'move_time_sec',
            'cmd_vx_mm_s',
            'cmd_vy_mm_s',
            'payload_abs_mm',
            'swing_mm',
            'swing_pitch_deg',
            'swing_roll_deg',
            'swing_axis_angle_deg',
            'id_filtered_swing_mm',
            'cart_q_mm',
            'cart_vx_mm_s',
            'cart_vy_mm_s',
            'gantry_state_stamp_s',
            'gantry_state_age_ms',
            'payload_sample_time_s',
            'payload_wall_age_ms',
            'payload_observed_clock_offset_s',
            'payload_calibrated_clock_offset_s',
            'payload_queue_delay_ms',
            'payload_encoder_sample_age_ms',
            'payload_rel_vx_filtered_mm_s',
            'payload_rel_vx_raw_mm_s',
            'stream_dt_ms',
            'traveled_mm',
            'cond_b',
            'omega_n_rad_s',
            'nonzero_ic_fitted_omega_rad_s',
            'nonzero_ic_small_angle_omega_rad_s',
            'nonzero_ic_fitted_amplitude_angle_deg',
            'nonzero_ic_frequency_correction_factor',
            'nonzero_ic_shaper_omega_rad_s',
            'nonzero_ic_shaper_frequency_scale',
            'nonzero_ic_shaper_frequency_mode',
            'nonzero_ic_configured_shaper_omega_rad_s',
            'nonzero_ic_robust_enabled',
            'nonzero_ic_robust_band_fraction',
            'nonzero_ic_robust_impulse_times_s',
            'nonzero_ic_robust_start_amplitudes',
            'nonzero_ic_robust_stop_amplitudes',
            'nonzero_ic_robust_worst_residual_fraction',
            'nonzero_ic_two_impulse_worst_residual_fraction',
            'nonzero_ic_robust_optimizer_iterations',
            'timed_profile_actuation_lead_ms',
            'timed_profile_fitted_peak_wall_s',
            'timed_profile_start_wall_s',
            'timed_profile_fitted_peak_payload_time_s',
            'timed_profile_start_payload_time_s',
            'timed_profile_payload_clock_offset_s',
            'timed_profile_arm_payload_queue_delay_ms',
            'nonzero_ic_predicted_peak_swing_mm',
            'nonzero_ic_predicted_peak_payload_velocity_mm_s',
            'nonzero_ic_predicted_command_start_swing_mm',
            'nonzero_ic_predicted_command_start_payload_velocity_mm_s',
            'nonzero_ic_measured_command_start_swing_mm',
            'nonzero_ic_measured_command_start_payload_velocity_mm_s',
            'nonzero_ic_measured_command_start_sample_offset_ms',
            'nonzero_ic_measured_peak_swing_mm',
            'nonzero_ic_measured_peak_payload_velocity_mm_s',
            'nonzero_ic_measured_peak_sample_offset_ms',
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
            'nonzero_ic_initial_swing_mm',
            'nonzero_ic_initial_payload_velocity_mm_s',
            'nonzero_ic_fit_amplitude_mm',
            'nonzero_ic_fit_rmse_mm',
            'nonzero_ic_fit_nrmse',
            'nonzero_ic_fit_samples',
            'nonzero_ic_fit_window_s',
            'id_candidate_time_s',
            'id_candidate_cond_b',
            'id_candidate_omega_n_rad_s',
            'id_candidate_zeta',
            'id_candidate_T_sec',
            'id_candidate_valid',
            'id_candidate_reject_reason',
            'eq21_input_model',
            'zero_zeta_T_sec',
            'zero_zeta_score',
            'zero_zeta_rmse',
            'zero_zeta_amp_mm',
            'final_id_source',
            'id_duration_s',
            'id_speed_mm_s',
            'two_mode_T2_sec',
            'two_mode_amp1_mm',
            'two_mode_amp2_mm',
            'two_mode_amp2_amp1',
            'two_mode_norm_rmse',
            'enc_pitch_count',
            'enc_roll_count',
            'enc_x_rel_mm',
            'enc_y_rel_mm',
            'enc_z_rel_mm',
            'enc_arduino_ms',
            'enc_pitch_raw',
            'enc_roll_raw',
            'enc_imu1_ax',
            'enc_imu1_ay',
            'enc_imu1_az',
            'enc_imu1_gx',
            'enc_imu1_gy',
            'enc_imu1_gz',
            'enc_imu2_ax',
            'enc_imu2_ay',
            'enc_imu2_az',
            'enc_imu2_gx',
            'enc_imu2_gy',
            'enc_imu2_gz',
            'enc_packet_age_ms',
            'enc_packet_seen',
            'enc_serial_lines',
            'enc_parse_errors',
            'enc_stale',
            'swing_cam_x_abs',
            'swing_cam_x_rel',
            'swing_cam_y_abs',
            'swing_cam_y_rel',
            'lat_state_stamp_s',
            'lat_encoder_read_stamp_s',
            'lat_stream_seq_rx',
            'lat_stream_seq_applied',
            'lat_source_stamp_s',
            'lat_controller_rx_stamp_s',
            'lat_apply_begin_stamp_s',
            'lat_apply_done_stamp_s',
            'lat_received_vx_mm_s',
            'lat_received_vy_mm_s',
            'lat_applied_vx_mm_s',
            'lat_applied_vy_mm_s',
            'lat_cart_x_mm',
            'lat_cart_y_mm',
            'lat_selected_vx_mm_s',
            'lat_selected_vy_mm_s',
            'lat_motor_vx_mm_s',
            'lat_motor_vy_mm_s',
            'lat_position_vx_mm_s',
            'lat_position_vy_mm_s',
            'lat_motor_a_position_rad',
            'lat_motor_b_position_rad',
            'lat_motor_a_velocity_rad_s',
            'lat_motor_b_velocity_rad_s',
            'lat_read_time_ms',
            'lat_write_time_ms',
            'lat_diagnostic_age_ms',
            'phase',
            'run_time_sec',
            'excite_angle_deg',
            'excite_angle_bias_deg',
            'excite_amplitude_est_deg',
            'excite_peak_est_deg',
            'excite_angle_rate_deg_s',
            'excite_cmd_vx_mm_s',
            'excite_drive_sign',
            'excite_cart_offset_mm',
            'excite_omega_rad_s',
        ])
        self.csv_file.flush()
        self.get_logger().info(f'CSV log: {path}')

    def _generate_run_outputs(self) -> None:
        if (
            self.run_outputs_generated
            or not self.args.auto_plot
            or self.csv_path is None
        ):
            return
        self.run_outputs_generated = True
        plotter = Path(__file__).resolve().with_name(
            'plot_adaptive_paper_run.py')
        if not plotter.exists():
            self.get_logger().warn(
                f'Automatic plotter is unavailable: {plotter}')
            return
        try:
            result = subprocess.run(
                [sys.executable, str(plotter), str(self.csv_path)],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
        except Exception as exc:
            self.get_logger().warn(f'Automatic plot generation failed: {exc}')
            return
        output_text = ' | '.join(
            line.strip()
            for line in (result.stdout + '\n' + result.stderr).splitlines()
            if line.strip()
        )
        if result.returncode == 0:
            self.get_logger().info(
                f'Automatic run outputs complete: {output_text}')
        else:
            self.get_logger().warn(
                f'Automatic run outputs failed ({result.returncode}): '
                f'{output_text}')

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
            self._auto_start_ready_wall = None
            now = time.monotonic()
            if now - self._last_wait_ready_print > 1.0:
                self._last_wait_ready_print = now
                self.get_logger().info(
                    'Waiting for fresh /payload/pose_e_rel and /gantry/state before start')
            return

        now = time.monotonic()
        if self.args.profile != 'nonzero-ic':
            start_delay_s = 0.0
        elif self.args.excite:
            # With --excite the trolley itself pumps the payload, so STREAM must
            # be live; the tau window instead follows swing-up completion.
            start_delay_s = self.args.excite_standclear_s
        else:
            start_delay_s = self.args.tau
        activation_label = (
            'closed-loop swing-up begins'
            if self.args.profile == 'nonzero-ic' and self.args.excite
            else 'Nonzero-IC TRAJ activates'
        )
        if self._auto_start_ready_wall is None:
            self._auto_start_ready_wall = now
            if start_delay_s > 0.0:
                self.get_logger().info(
                    f'Telemetry ready. {activation_label} in '
                    f'{start_delay_s:.1f}s; prepare the payload and stand clear.')
        remaining_s = start_delay_s - (now - self._auto_start_ready_wall)
        if remaining_s > 0.0:
            if now - self._last_wait_ready_print > 0.9:
                self._last_wait_ready_print = now
                self.get_logger().info(
                    f'{activation_label} in {remaining_s:.1f}s')
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
        now = time.monotonic()
        if (
            self.latest_gantry_state_sample is None
            or now - self.latest_gantry_state_sample.rx_wall
            > self.args.gantry_fresh_timeout
        ):
            return False
        if self.latest_payload_abs_mm is None or self.latest_payload_wall is None:
            return False
        if now - self.latest_payload_wall > self.args.payload_fresh_timeout:
            return False
        if (
            self.args.profile == 'nonzero-ic'
            and self.args.require_encoder_health
            and not self._encoder_health_ready(now)
        ):
            return False
        if self.args.profile == 'nonzero-ic' and self.args.excite:
            angle = self._excite_axis_angle_deg()
            if angle is None or not math.isfinite(angle):
                return False
            if (
                self.latest_enc_wall is None
                or now - self.latest_enc_wall
                > self.args.payload_fresh_timeout
            ):
                return False
        if (
            self.args.profile == 'nonzero-ic'
            and not self.args.nonzero_ic_adaptive_frequency
            and self.args.initial_payload_velocity_mm_s is None
        ):
            relative_velocity = (
                self.latest_payload_rel_vx_mm_s
                if self.args.axis == 'x'
                else self.latest_payload_rel_vy_mm_s
            )
            if relative_velocity is None or not math.isfinite(relative_velocity):
                return False
        return True

    def _on_enable_done(self, future):
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warn(f'/gantry/enable failed: {exc}')
            return
        if result is not None:
            self.get_logger().info(f'/gantry/enable: {result.message}')

    def _msg_stamp_sec(self, msg: GantryState) -> float:
        stamp = msg.header.stamp
        return float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)

    def _gantry_state_cb(self, msg: GantryState):
        self.latest_gantry_state = msg
        rx_wall = time.monotonic()
        sample = GantryStateSample(
            rx_wall=rx_wall,
            stamp=self._msg_stamp_sec(msg),
            x=float(msg.x),
            y=float(msg.y),
            vx=float(msg.vx),
            vy=float(msg.vy),
        )
        self.latest_gantry_state_sample = sample
        self.gantry_state_samples.append(sample)
        cutoff = rx_wall - 2.0
        self.gantry_state_samples = [
            old for old in self.gantry_state_samples if old.rx_wall >= cutoff
        ]

    def _cart_state_for_wall_time(self, wall_t: float) -> GantryStateSample | None:
        samples = self.gantry_state_samples
        if not samples:
            return None
        if wall_t <= samples[0].rx_wall:
            return samples[0]
        if wall_t >= samples[-1].rx_wall:
            return samples[-1]
        for prev, nxt in zip(samples[:-1], samples[1:]):
            if prev.rx_wall <= wall_t <= nxt.rx_wall:
                dt = nxt.rx_wall - prev.rx_wall
                if dt <= 1.0e-9:
                    return nxt
                a = (wall_t - prev.rx_wall) / dt
                return GantryStateSample(
                    rx_wall=wall_t,
                    stamp=prev.stamp + a * (nxt.stamp - prev.stamp),
                    x=prev.x + a * (nxt.x - prev.x),
                    y=prev.y + a * (nxt.y - prev.y),
                    vx=prev.vx + a * (nxt.vx - prev.vx),
                    vy=prev.vy + a * (nxt.vy - prev.vy),
                )
        return samples[-1]

    def _traj_cmd_cb(self, msg: TrajCmd):
        if msg.command != TrajCmd.MOTION_START or self.motion_started:
            return
        self.motion_started = True
        self.run_start_wall = time.monotonic()
        if self.args.profile == 'nonzero-ic':
            # The TRAJ controller is active, but keep streaming zero until the
            # freely swinging payload reaches a state whose closed-form
            # solution has 0 <= A0,A1 <= 1.  Profile time begins only when that
            # state is captured, so no reverse command is ever needed.
            self.wall_start = None
            self.schedule = None
            if self.args.excite and self.exciter is not None:
                self.phase = 'excite'
                self.excite_start_cart_q_mm = self.latest_cart_q_mm
                # The bounded-cycle controller estimates amplitude from each
                # cycle's peak-to-peak angle.  Do not reinterpret the first
                # instantaneous sample as vertical; the previous hardware run
                # began at +3.63 deg and that false bias destabilized feedback.
                self.excite_angle_bias_deg = 0.0
                self.get_logger().info(
                    'MOTION_START received; beginning bounded positive-axis '
                    f'swing-up to {self.args.excite_target_angle_deg:.1f}deg; '
                    f'anchor={self.excite_start_cart_q_mm:.1f}mm.')
            else:
                self.phase = 'wait_peak'
                self.get_logger().info(
                    'MOTION_START received; holding the trolley at zero velocity '
                    'until a forward-only nonzero-IC solution is available')
            return
        self.phase = 'maneuver'
        self.wall_start = time.monotonic()
        if (
            self.args.profile != 'nonzero-ic'
            and self.latest_payload_abs_mm is not None
            and self.latest_payload_time is not None
        ):
            self._begin_id()
        self.get_logger().info('MOTION_START: paper TDF move started')

    def _encoder_pose_cb(self, msg: Float64MultiArray):
        # /payload/pose_e data:
        # [time, pitch_deg, roll_deg, pitch_count, roll_count]
        d = msg.data
        if len(d) >= 5:
            self.latest_enc_time = float(d[0])
            self.latest_enc_pitch_deg = float(d[1])
            self.latest_enc_roll_deg = float(d[2])
            self.latest_enc_pitch_count = float(d[3])
            self.latest_enc_roll_count = float(d[4])
            self.latest_enc_wall = time.monotonic()

    def _encoder_diag_cb(self, msg: Float64MultiArray):
        # /payload/encoder/diagnostics data:
        # [time, arduino_ms, pitch_raw, roll_raw, pitch_count, roll_count,
        #  optional imu1/imu2 accel/gyro values,
        #  optional packet_age_ms, packet_seen, serial_lines, parse_errors, stale]
        d = msg.data
        self.latest_enc_diag_wall = time.monotonic()
        fields = []
        if msg.layout.dim and msg.layout.dim[0].label:
            fields = [name.strip() for name in msg.layout.dim[0].label.split(',')]
        if fields and len(fields) <= len(d):
            values = {name: float(d[index]) for index, name in enumerate(fields)}

            def value(*names):
                for name in names:
                    if name in values:
                        return values[name]
                return None

            self.latest_enc_time = value('time')
            self.latest_enc_arduino_ms = value('arduino_ms')
            self.latest_enc_pitch_raw = value('pitch_raw')
            self.latest_enc_roll_raw = value('roll_raw')
            self.latest_enc_pitch_count = value('pitch_count')
            self.latest_enc_roll_count = value('roll_count')
            self.latest_enc_imu1_ax = value('imu1_ax', 'imu1_ax_g')
            self.latest_enc_imu1_ay = value('imu1_ay', 'imu1_ay_g')
            self.latest_enc_imu1_az = value('imu1_az', 'imu1_az_g')
            self.latest_enc_imu1_gx = value('imu1_gx', 'imu1_gx_dps')
            self.latest_enc_imu1_gy = value('imu1_gy', 'imu1_gy_dps')
            self.latest_enc_imu1_gz = value('imu1_gz', 'imu1_gz_dps')
            self.latest_enc_imu2_ax = value('imu2_ax', 'imu2_ax_g')
            self.latest_enc_imu2_ay = value('imu2_ay', 'imu2_ay_g')
            self.latest_enc_imu2_az = value('imu2_az', 'imu2_az_g')
            self.latest_enc_imu2_gx = value('imu2_gx', 'imu2_gx_dps')
            self.latest_enc_imu2_gy = value('imu2_gy', 'imu2_gy_dps')
            self.latest_enc_imu2_gz = value('imu2_gz', 'imu2_gz_dps')
            self.latest_enc_packet_age_ms = value('packet_age_ms')
            self.latest_enc_packet_seen = value('packet_seen')
            self.latest_enc_serial_lines = value('serial_lines')
            self.latest_enc_parse_errors = value('parse_errors')
            self.latest_enc_stale = value('stale')
            return
        if len(d) >= 9:
            self.latest_enc_time = float(d[0])
            self.latest_enc_arduino_ms = float(d[1])
            self.latest_enc_pitch_raw = float(d[2])
            self.latest_enc_roll_raw = float(d[3])
            self.latest_enc_pitch_count = float(d[4])
            self.latest_enc_roll_count = float(d[5])
            if len(d) >= 21:
                self.latest_enc_imu1_ax = float(d[6])
                self.latest_enc_imu1_ay = float(d[7])
                self.latest_enc_imu1_az = float(d[8])
                self.latest_enc_imu1_gx = float(d[9])
                self.latest_enc_imu1_gy = float(d[10])
                self.latest_enc_imu1_gz = float(d[11])
                self.latest_enc_imu2_ax = float(d[12])
                self.latest_enc_imu2_ay = float(d[13])
                self.latest_enc_imu2_az = float(d[14])
                self.latest_enc_imu2_gx = float(d[15])
                self.latest_enc_imu2_gy = float(d[16])
                self.latest_enc_imu2_gz = float(d[17])
                if len(d) >= 23:
                    self.latest_enc_packet_age_ms = float(d[18])
                    self.latest_enc_packet_seen = float(d[19])
                    self.latest_enc_serial_lines = float(d[20])
                    self.latest_enc_parse_errors = float(d[21])
                    self.latest_enc_stale = float(d[22])
                else:
                    self.latest_enc_serial_lines = float(d[18])
                    self.latest_enc_parse_errors = float(d[19])
                    self.latest_enc_stale = float(d[20])
            else:
                self.latest_enc_serial_lines = float(d[6])
                self.latest_enc_parse_errors = float(d[7])
                self.latest_enc_stale = float(d[8])

    def _encoder_health_ready(self, now: float | None = None) -> bool:
        """True only when the diagnostic stream explicitly reports healthy."""
        check_wall = time.monotonic() if now is None else now
        return (
            self.latest_enc_diag_wall is not None
            and check_wall - self.latest_enc_diag_wall
            <= self.args.payload_fresh_timeout
            and self.latest_enc_stale is not None
            and math.isfinite(self.latest_enc_stale)
            and self.latest_enc_stale < 0.5
        )

    def _payload_state_cb(self, msg: PayloadState):
        # gantry_x/gantry_y are already rotated + sign-corrected from camera
        # optical frame into gantry (cart) axes — see payload_frames.py.
        self.latest_cam_gantry_x_mm = 1000.0 * float(msg.gantry_x)
        self.latest_cam_gantry_y_mm = 1000.0 * float(msg.gantry_y)

    def _traj_latency_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 26:
            return
        self.latest_traj_latency = [float(value) for value in msg.data[:26]]
        self.latest_traj_latency_wall = time.monotonic()

    def _capture_timed_payload_observations(
        self,
        cart_sample: GantryStateSample | None,
    ) -> None:
        """Record measured states nearest command start and physical peak.

        Event offsets use ``/payload/pose_e_rel``'s own monotonic time base.
        The velocity is the publisher's filtered relative-velocity estimate
        plus measured cart velocity, so it is deliberately kept separate from
        the sinusoid-projected state used to construct the shaper.
        """
        if (
            self.latest_payload_time is None
            or self.latest_swing_mm is None
            or cart_sample is None
        ):
            return
        raw_velocity_mm_s = (
            getattr(self, 'latest_payload_rel_raw_vx_mm_s', None)
            if self.args.axis == 'x'
            else getattr(self, 'latest_payload_rel_raw_vy_mm_s', None)
        )
        relative_velocity_mm_s = (
            raw_velocity_mm_s
            if raw_velocity_mm_s is not None
            and math.isfinite(raw_velocity_mm_s)
            else (
                self.latest_payload_rel_vx_mm_s
                if self.args.axis == 'x'
                else self.latest_payload_rel_vy_mm_s
            )
        )
        if relative_velocity_mm_s is None:
            return
        cart_velocity_mm_s = 1000.0 * float(
            cart_sample.vx if self.args.axis == 'x' else cart_sample.vy
        )
        world_velocity_mm_s = relative_velocity_mm_s + cart_velocity_mm_s
        if not math.isfinite(world_velocity_mm_s):
            return

        values = (
            (
                'nonzero_ic_measured_command_start',
                self.timed_profile_start_payload_time,
            ),
            ('nonzero_ic_measured_peak', self.timed_profile_peak_payload_time),
        )
        for attribute, target_time_s in values:
            if target_time_s is None:
                continue
            current = getattr(self, attribute)
            setattr(
                self,
                attribute,
                closer_timed_payload_observation(
                    current,
                    sample_time_s=self.latest_payload_time,
                    target_time_s=target_time_s,
                    swing_mm=self.latest_swing_mm,
                    world_payload_velocity_mm_s=world_velocity_mm_s,
                ),
            )

    def _payload_cb(self, msg: Float64MultiArray):
        index = 3 if self.args.axis == 'x' else 4
        if len(msg.data) <= index:
            self.get_logger().warn('Encoder relative array too short')
            return
        payload_wall = time.monotonic()

        self.latest_payload_rel_x_mm = 1000.0 * float(msg.data[3]) if len(msg.data) > 3 else None
        self.latest_payload_rel_y_mm = 1000.0 * float(msg.data[4]) if len(msg.data) > 4 else None
        self.latest_payload_rel_z_mm = 1000.0 * float(msg.data[5]) if len(msg.data) > 5 else None
        self.latest_payload_rel_vx_mm_s = (
            1000.0 * float(msg.data[6])
            if len(msg.data) > 6 and math.isfinite(float(msg.data[6]))
            else None
        )
        self.latest_payload_rel_vy_mm_s = (
            1000.0 * float(msg.data[7])
            if len(msg.data) > 7 and math.isfinite(float(msg.data[7]))
            else None
        )
        fields = []
        if msg.layout.dim and msg.layout.dim[0].label:
            fields = [
                name.strip() for name in msg.layout.dim[0].label.split(',')
            ]

        def optional_field(name: str) -> float | None:
            if name not in fields:
                return None
            field_index = fields.index(name)
            if field_index >= len(msg.data):
                return None
            value = float(msg.data[field_index])
            return value if math.isfinite(value) else None

        raw_vx_m_s = optional_field('vx_rel_raw_m_s')
        raw_vy_m_s = optional_field('vy_rel_raw_m_s')
        self.latest_payload_rel_raw_vx_mm_s = (
            None if raw_vx_m_s is None else 1000.0 * raw_vx_m_s
        )
        self.latest_payload_rel_raw_vy_mm_s = (
            None if raw_vy_m_s is None else 1000.0 * raw_vy_m_s
        )
        self.latest_payload_encoder_sample_age_ms = optional_field(
            'sample_age_ms')

        swing_m = float(msg.data[index])
        cart_sample = self._cart_state_for_wall_time(payload_wall)
        self.latest_cart_state_sample = cart_sample
        if self.latest_gantry_state_sample is None:
            self.latest_gantry_state_age_ms = None
        else:
            self.latest_gantry_state_age_ms = (
                1000.0 * max(0.0, payload_wall - self.latest_gantry_state_sample.rx_wall)
            )
        cart_m = self._current_cart_axis_m(cart_sample)
        if cart_m is None:
            cart_m = 0.0

        payload_time = float(msg.data[0])
        if math.isfinite(payload_time):
            (
                self.payload_clock_offset_s,
                payload_queue_delay_s,
                payload_clock_reset,
            ) = update_payload_clock_calibration(
                self.payload_clock_offset_s,
                self.latest_payload_time,
                payload_time,
                payload_wall,
            )
            self.latest_payload_observed_clock_offset_s = (
                payload_wall - payload_time
            )
            self.latest_payload_queue_delay_ms = (
                1000.0 * payload_queue_delay_s
            )
            if payload_clock_reset:
                self.get_logger().warn(
                    'Payload publisher timestamp reset detected; restarted '
                    'payload-clock calibration')
        else:
            self.latest_payload_observed_clock_offset_s = None
            self.latest_payload_queue_delay_ms = None

        self.latest_payload_time = payload_time
        self.latest_payload_wall = payload_wall
        self.latest_payload_age_ms = 0.0
        self.latest_swing_mm = swing_m * 1000.0
        self.latest_cart_q_mm = cart_m * 1000.0
        self.latest_payload_abs_mm = (cart_m + swing_m) * 1000.0
        self._capture_timed_payload_observations(cart_sample)

        if (
            self.args.profile == 'nonzero-ic'
            and self.wall_start is None
            and (not self.args.excite or self.id_hold_start_wall is not None)
            and math.isfinite(self.latest_payload_time)
            and math.isfinite(self.latest_swing_mm)
        ):
            self.nonzero_ic_swing_samples.append((
                self.latest_payload_time,
                self.latest_swing_mm,
            ))
            history_s = max(self.args.tau + 1.0, 1.25 * self.args.tau)
            cutoff_time = self.latest_payload_time - history_s
            self.nonzero_ic_swing_samples = [
                sample
                for sample in self.nonzero_ic_swing_samples
                if sample[0] >= cutoff_time
            ]

        if self.final_zero_wall is not None:
            residual_t = self.latest_payload_wall - self.final_zero_wall
            if residual_t >= 0.0:
                self.residual_samples.append((residual_t, self.latest_swing_mm))

        if (
            self.args.profile != 'nonzero-ic'
            and self.motion_started
            and self.identifier.t0 is None
        ):
            self._begin_id()
        if not self.motion_started or self.identifier.t0 is None:
            return

        # Once the schedule is fixed, estimator updates cannot affect control.
        # Keep this callback lightweight so payload and residual logging retain
        # their full sample rate for the remainder of the run.
        if self.schedule is not None:
            if not self._id_updates_stopped_logged:
                self._id_updates_stopped_logged = True
                self.get_logger().info(
                    'Online ID updates stopped after schedule lock; '
                    'payload and residual logging continue')
            return

        payload_abs_for_id = self._payload_abs_for_id()
        q_for_eq21 = (
            None
            if self.args.eq21_input_model == 'ideal_k'
            else self.latest_cart_q_mm
        )
        if self.args.id_method == 'integral':
            est = self.identifier.update(
                self.latest_payload_time,
                payload_abs_for_id,
                q_for_eq21,
            )
            if est is not None:
                self._publish_estimate(est)
                self._record_id_report_estimate(est)
                self._maybe_lock_from_estimate(est)

        paper2_est = self._update_paper2_step_estimate()
        if paper2_est is not None and self.args.id_method == 'paper2-step':
            self._publish_estimate(paper2_est)
            self._record_id_report_estimate(paper2_est)
            self._maybe_lock_from_estimate(paper2_est)

        freq_est = self._update_freq_bank_estimate()
        if freq_est is not None and self.args.id_method == 'freq-bank':
            self._publish_estimate(freq_est)
            self._record_id_report_estimate(freq_est)
            self._maybe_lock_from_estimate(freq_est)

        zero_est = self._update_zero_zeta_estimate()
        if zero_est is not None and self.args.id_method == 'zero-zeta-ls':
            self._publish_estimate(zero_est)
            self._record_id_report_estimate(zero_est)
            self._maybe_lock_from_estimate(zero_est)

    def _current_cart_axis_m(
        self, sample: GantryStateSample | None = None
    ) -> float | None:
        if sample is not None:
            return sample.x if self.args.axis == 'x' else sample.y
        if self.latest_gantry_state is None:
            return None
        return (
            float(self.latest_gantry_state.x)
            if self.args.axis == 'x'
            else float(self.latest_gantry_state.y)
        )

    def _payload_abs_for_id(self) -> float:
        if (
            self.args.id_lowpass_hz <= 0.0
            or self.latest_payload_time is None
            or self.latest_swing_mm is None
            or self.latest_cart_q_mm is None
        ):
            return 0.0 if self.latest_payload_abs_mm is None else self.latest_payload_abs_mm

        raw_swing = self.latest_swing_mm
        prev = self.latest_id_filtered_swing_mm
        prev_t = self._id_filter_time
        if prev is None or prev_t is None:
            filtered = raw_swing
        else:
            dt = max(0.0, self.latest_payload_time - prev_t)
            alpha = 1.0 - math.exp(-2.0 * math.pi * self.args.id_lowpass_hz * dt)
            alpha = max(0.0, min(alpha, 1.0))
            filtered = prev + alpha * (raw_swing - prev)

        self.latest_id_filtered_swing_mm = filtered
        self._id_filter_time = self.latest_payload_time
        return self.latest_cart_q_mm + filtered

    def _begin_id(self):
        if self.latest_payload_time is None or self.latest_payload_abs_mm is None:
            return
        q = 0.0 if self.latest_cart_q_mm is None else self.latest_cart_q_mm
        if self.start_cart_q_mm is None:
            self.start_cart_q_mm = q
        self.latest_id_filtered_swing_mm = None
        self._id_filter_time = None
        self.mode_fit_samples = []
        self.zero_zeta_samples = []
        self.latest_zero_zeta_estimate = None
        self.latest_zero_zeta_score = None
        self.latest_zero_zeta_rmse = None
        self.latest_zero_zeta_amp = None
        self.latest_freq_bank_estimate = None
        self.latest_paper2_estimate = None
        self.latest_two_mode_estimate = None
        self.latest_two_mode_t2 = None
        self.latest_two_mode_amp1 = None
        self.latest_two_mode_amp2 = None
        self.latest_two_mode_amp_ratio = None
        self.latest_two_mode_nrmse = None
        self._last_mode_fit_log_t = -1.0
        self._last_zero_zeta_update_t = -1.0
        self._zero_zeta_future = None
        self._last_freq_bank_update_t = -1.0
        self._last_two_mode_update_t = -1.0
        payload_abs_for_id = self._payload_abs_for_id()
        self.identifier.start(
            self.latest_payload_time,
            payload_abs_for_id,
            q_now_mm=q,
            zero_at_start=True,
        )
        self.get_logger().info(
            f'ID started: x={payload_abs_for_id:.2f} mm q={q:.2f} mm')

    @staticmethod
    def _fit_zero_zeta_grid(
        t: float,
        x: float,
        swing_p2p: float,
        times: np.ndarray,
        swing: np.ndarray,
        t_min: float,
        t_max: float,
        grid_count: int,
        min_amp: float,
        max_score: float,
    ) -> tuple[float, float, float, float, float, float, float, float, float] | None:
        """Fit all period candidates off the ROS callback thread."""
        tc = times - float(times[0])
        periods = np.linspace(t_min, t_max, max(5, grid_count))
        omegas = math.pi / periods
        phase = omegas[:, None] * tc[None, :]
        design = np.empty((len(periods), len(tc), 4), dtype=float)
        design[:, :, 0] = 1.0
        design[:, :, 1] = tc
        design[:, :, 2] = np.cos(phase)
        design[:, :, 3] = np.sin(phase)

        try:
            xtx = np.einsum('gni,gnj->gij', design, design, optimize=True)
            xty = np.einsum('gni,n->gi', design, swing, optimize=True)
            coefs = np.linalg.solve(xtx, xty)
            pred = np.einsum('gni,gi->gn', design, coefs, optimize=True)
            rmse = np.sqrt(np.mean((swing[None, :] - pred) ** 2, axis=1))
            amp = np.hypot(coefs[:, 2], coefs[:, 3])
            score = rmse / np.maximum(amp, 1.0e-9)
            cond = np.sqrt(np.linalg.cond(xtx))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return None

        valid = (
            np.isfinite(score)
            & np.isfinite(rmse)
            & np.isfinite(amp)
            & np.isfinite(cond)
            & (amp >= min_amp)
            & (score <= max_score)
        )
        if not np.any(valid):
            return None
        valid_indices = np.flatnonzero(valid)
        best_index = int(valid_indices[np.argmin(score[valid])])
        return (
            t,
            x,
            swing_p2p,
            float(score[best_index]),
            float(periods[best_index]),
            float(omegas[best_index]),
            float(cond[best_index]),
            float(rmse[best_index]),
            float(amp[best_index]),
        )

    def _update_zero_zeta_estimate(self) -> Estimate | None:
        if self.args.id_method != 'zero-zeta-ls':
            return None
        if (
            self.identifier.t0 is None
            or self.latest_payload_time is None
            or self.latest_swing_mm is None
        ):
            return None

        t = self.latest_payload_time - self.identifier.t0
        if t <= 0.0:
            return None

        self.zero_zeta_samples.append((t, self.latest_swing_mm))
        keep_after = max(
            0.0,
            t - max(self.args.zero_zeta_history_s, self.args.zero_zeta_window_s),
        )
        self.zero_zeta_samples = [
            sample for sample in self.zero_zeta_samples if sample[0] >= keep_after
        ]

        completed = None
        if self._zero_zeta_future is not None and self._zero_zeta_future.done():
            try:
                completed = self._zero_zeta_future.result()
            except Exception as exc:
                self.get_logger().error(f'Zero-zeta ID worker failed: {exc}')
            self._zero_zeta_future = None

        est = None
        if completed is not None:
            fit_t, fit_x, swing_p2p, score, T, omega, cond, rmse, amp = completed
            est = Estimate(
                t=fit_t,
                x=fit_x,
                i1=float('nan'),
                i2=float('nan'),
                i3=float('nan'),
                cond_b=score,
                omega_n=omega,
                zeta=0.0,
                T=T,
                A0=float('nan'),
                A1=float('nan'),
                A2=float('nan'),
            )
            self.latest_zero_zeta_estimate = est
            self.latest_zero_zeta_score = score
            self.latest_zero_zeta_rmse = rmse
            self.latest_zero_zeta_amp = amp
            if fit_t - self._last_mode_fit_log_t >= self.args.print_period:
                self._last_mode_fit_log_t = fit_t
                self.get_logger().info(
                    f'zero-zeta LS ID: t={fit_t:.3f}s T={T:.4f}s '
                    f'omega={omega:.4f} amp={amp:.2f}mm p2p={swing_p2p:.2f}mm '
                    f'nrmse={score:.3f} rmse={rmse:.3f} cond={cond:.3e}')

        if t < self.args.zero_zeta_after_s:
            return est
        if self._zero_zeta_future is not None:
            return est
        if t - self._last_zero_zeta_update_t < self.args.zero_zeta_update_period_s:
            return est
        self._last_zero_zeta_update_t = t

        window_start = (
            0.0
            if self.args.zero_zeta_window_s <= 0.0
            else max(0.0, t - self.args.zero_zeta_window_s)
        )
        window = [
            sample
            for sample in self.zero_zeta_samples
            if window_start <= sample[0] <= t
        ]
        if len(window) < self.args.zero_zeta_min_samples:
            return est

        times = np.array([sample[0] for sample in window], dtype=float)
        swing = np.array([sample[1] for sample in window], dtype=float)
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(swing)):
            return est

        swing_p2p = float(np.max(swing) - np.min(swing))
        if swing_p2p < self.args.zero_zeta_min_p2p_mm:
            return est

        if self._zero_zeta_executor is not None:
            self._zero_zeta_future = self._zero_zeta_executor.submit(
                self._fit_zero_zeta_grid,
                t,
                float(self.latest_swing_mm),
                swing_p2p,
                times,
                swing,
                self.args.zero_zeta_t_min,
                self.args.zero_zeta_t_max,
                self.args.zero_zeta_grid_count,
                self.args.zero_zeta_min_amp_mm,
                self.args.zero_zeta_max_norm_rmse,
            )
        return est

    def _update_freq_bank_estimate(self) -> Estimate | None:
        if self.args.id_method != 'freq-bank':
            return None
        if (
            self.identifier.t0 is None
            or self.latest_payload_time is None
            or self.latest_swing_mm is None
        ):
            return None

        t = self.latest_payload_time - self.identifier.t0
        if t <= 0.0:
            return None

        self.mode_fit_samples.append((t, self.latest_swing_mm))
        keep_after = max(0.0, t - max(self.args.mode_fit_history_s, self.args.mode_fit_window_s))
        self.mode_fit_samples = [
            sample for sample in self.mode_fit_samples if sample[0] >= keep_after
        ]
        if t < self.args.mode_fit_after_s:
            return None
        if t - self._last_freq_bank_update_t < self.args.freq_bank_update_period_s:
            return None
        self._last_freq_bank_update_t = t

        window_start = max(0.0, t - self.args.mode_fit_window_s)
        window = [
            sample
            for sample in self.mode_fit_samples
            if window_start <= sample[0] <= t
        ]
        if len(window) < self.args.mode_fit_min_samples:
            return None

        times = np.array([sample[0] for sample in window], dtype=float)
        swing = np.array([sample[1] for sample in window], dtype=float)
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(swing)):
            return None

        swing_p2p = float(np.max(swing) - np.min(swing))
        if swing_p2p < self.args.mode_fit_min_p2p_mm:
            return None

        tau = times - float(times[0])
        y = swing.astype(float)
        # Remove constant and slow drift so coherent oscillation dominates the bank.
        try:
            trend_design = np.column_stack((np.ones_like(tau), tau))
            trend_coef, _, _, _ = np.linalg.lstsq(trend_design, y, rcond=None)
            y = y - trend_design @ trend_coef
        except np.linalg.LinAlgError:
            y = y - float(np.mean(y))
        y = y - float(np.mean(y))

        y_energy = float(np.dot(y, y))
        if y_energy <= 1.0e-9:
            return None

        weights = np.hanning(len(y))
        if not np.any(weights > 0.0):
            weights = np.ones_like(y)
        yw = y * weights
        best: tuple[float, float, float, float, float] | None = None
        powers: list[float] = []
        n_grid = max(5, self.args.mode_fit_grid_count)
        for T in np.linspace(self.args.mode_fit_t_min, self.args.mode_fit_t_max, n_grid):
            if T <= 0.0:
                continue
            omega = math.pi / float(T)
            phase = omega * (times - float(times[0]))
            c = np.cos(phase) * weights
            s = np.sin(phase) * weights
            cc = float(np.dot(c, c))
            ss = float(np.dot(s, s))
            if cc <= 1.0e-12 or ss <= 1.0e-12:
                continue
            ac = float(np.dot(yw, c)) / cc
            bs = float(np.dot(yw, s)) / ss
            pred = ac * np.cos(phase) + bs * np.sin(phase)
            amp = float(math.hypot(ac, bs))
            if amp < self.args.mode_fit_min_amp_mm:
                continue
            err = y - pred
            rmse = float(np.sqrt(np.mean(err * err)))
            norm_rmse = rmse / max(amp, 1.0e-9)
            if norm_rmse > self.args.mode_fit_max_norm_rmse:
                continue
            # Normalized coherent power in [0, ~1]; larger means stronger frequency support.
            power = float((ac * ac * cc + bs * bs * ss) / max(y_energy, 1.0e-9))
            if not all(math.isfinite(v) for v in (power, amp, rmse, norm_rmse)):
                continue
            powers.append(power)
            if best is None or power > best[0]:
                best = (power, float(T), omega, amp, norm_rmse)

        if best is None or not powers:
            return None

        power, T, omega, amp, norm_rmse = best
        powers_sorted = sorted(powers, reverse=True)
        second = powers_sorted[1] if len(powers_sorted) > 1 else 1.0e-12
        ratio = power / max(second, 1.0e-12)
        if ratio < self.args.freq_bank_min_ratio:
            return None
        cond_b = 1.0 / max(power * max(ratio - 1.0, 1.0e-3), 1.0e-6)

        est = Estimate(
            t=t,
            x=self.latest_swing_mm,
            i1=float('nan'),
            i2=float('nan'),
            i3=float('nan'),
            cond_b=cond_b,
            omega_n=omega,
            zeta=0.0,
            T=T,
            A0=float('nan'),
            A1=float('nan'),
            A2=float('nan'),
        )
        self.latest_freq_bank_estimate = est
        if t - self._last_mode_fit_log_t >= self.args.print_period:
            self._last_mode_fit_log_t = t
            self.get_logger().info(
                f'freq-bank ID: t={t:.3f}s T={T:.4f}s omega={omega:.4f} '
                f'power={power:.3f} ratio={ratio:.3f} condB={cond_b:.3e} '
                f'amp={amp:.2f}mm p2p={swing_p2p:.2f}mm nrmse={norm_rmse:.3f}')
        return est

    def _update_two_mode_fit_estimate(self) -> Estimate | None:
        if self.args.id_method != 'two-mode-fit':
            return None
        if (
            self.identifier.t0 is None
            or self.latest_payload_time is None
            or self.latest_swing_mm is None
        ):
            return None

        t = self.latest_payload_time - self.identifier.t0
        if t <= 0.0:
            return None
        self.mode_fit_samples.append((t, self.latest_swing_mm))
        keep_after = max(0.0, t - max(self.args.two_mode_history_s, self.args.two_mode_window_s))
        self.mode_fit_samples = [
            sample for sample in self.mode_fit_samples if sample[0] >= keep_after
        ]
        if t - self._last_two_mode_update_t < self.args.two_mode_update_period_s:
            return None
        self._last_two_mode_update_t = t
        if t < self.args.two_mode_after_s:
            return None

        window_start = max(0.0, t - self.args.two_mode_window_s)
        window = [
            sample
            for sample in self.mode_fit_samples
            if window_start <= sample[0] <= t
        ]
        if len(window) < self.args.two_mode_min_samples:
            return None

        times = np.array([sample[0] for sample in window], dtype=float)
        swing = np.array([sample[1] for sample in window], dtype=float)
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(swing)):
            return None

        swing_p2p = float(np.max(swing) - np.min(swing))
        if swing_p2p < self.args.mode_fit_min_p2p_mm:
            return None

        t_center = times - float(np.mean(times))
        y = swing - float(np.mean(swing))
        best: tuple[float, float, float, float, float, float, float, float] | None = None
        t1_values = np.linspace(
            self.args.two_mode_t1_min,
            self.args.two_mode_t1_max,
            max(5, self.args.two_mode_t1_grid_count),
        )
        t2_values = np.linspace(
            self.args.two_mode_t2_min,
            self.args.two_mode_t2_max,
            max(5, self.args.two_mode_t2_grid_count),
        )
        for T1 in t1_values:
            if T1 <= 0.0:
                continue
            omega1 = math.pi / float(T1)
            c1 = np.cos(omega1 * t_center)
            s1 = np.sin(omega1 * t_center)
            for T2 in t2_values:
                if T2 <= 0.0 or T2 >= self.args.two_mode_max_t2_t1_ratio * T1:
                    continue
                omega2 = math.pi / float(T2)
                design = np.column_stack(
                    (
                        np.ones_like(t_center),
                        t_center,
                        c1,
                        s1,
                        np.cos(omega2 * t_center),
                        np.sin(omega2 * t_center),
                    )
                )
                try:
                    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
                    cond = float(np.linalg.cond(design))
                except np.linalg.LinAlgError:
                    continue
                if rank < 6:
                    continue
                pred = design @ coef
                err = y - pred
                rmse = float(np.sqrt(np.mean(err * err)))
                amp1 = float(math.hypot(float(coef[2]), float(coef[3])))
                amp2 = float(math.hypot(float(coef[4]), float(coef[5])))
                if amp1 < self.args.two_mode_min_amp1_mm:
                    continue
                norm_rmse = rmse / max(amp1, 1.0e-9)
                if norm_rmse > self.args.two_mode_max_norm_rmse:
                    continue
                if not all(math.isfinite(v) for v in (cond, rmse, amp1, amp2, norm_rmse)):
                    continue
                score = (
                    norm_rmse
                    + self.args.two_mode_cond_weight * math.log10(max(cond, 1.0))
                    + self.args.two_mode_amp2_weight * amp2 / max(amp1, 1.0e-9)
                )
                if best is None or score < best[0]:
                    best = (score, float(T1), float(T2), cond, rmse, amp1, amp2, norm_rmse)

        if best is None:
            return None

        score, T1, T2, cond, _rmse, amp1, amp2, norm_rmse = best
        omega1 = math.pi / T1
        amp_ratio = amp2 / max(amp1, 1.0e-9)
        est = Estimate(
            t=t,
            x=self.latest_swing_mm,
            i1=float('nan'),
            i2=float('nan'),
            i3=float('nan'),
            cond_b=cond,
            omega_n=omega1,
            zeta=0.0,
            T=T1,
            A0=float('nan'),
            A1=amp1,
            A2=amp2,
        )
        self.latest_two_mode_estimate = est
        self.latest_two_mode_t2 = T2
        self.latest_two_mode_amp1 = amp1
        self.latest_two_mode_amp2 = amp2
        self.latest_two_mode_amp_ratio = amp_ratio
        self.latest_two_mode_nrmse = norm_rmse
        if t - self._last_mode_fit_log_t >= self.args.print_period:
            self._last_mode_fit_log_t = t
            self.get_logger().info(
                f'two-mode-fit ID: t={t:.3f}s T1={T1:.4f}s T2={T2:.4f}s '
                f'amp1={amp1:.2f}mm amp2={amp2:.2f}mm ratio={amp_ratio:.3f} '
                f'nrmse={norm_rmse:.3f} cond={cond:.3e} score={score:.3f}')
        return est

    def _update_paper2_step_estimate(self) -> Estimate | None:
        if self.args.id_method != 'paper2-step':
            return None
        if (
            self.identifier.t0 is None
            or self.latest_payload_time is None
            or self.latest_swing_mm is None
        ):
            return None

        t = self.latest_payload_time - self.identifier.t0
        if t <= 0.0:
            return None
        if not self.mode_fit_samples or self.mode_fit_samples[-1][0] < t:
            self.mode_fit_samples.append((t, self.latest_swing_mm))
        keep_after = max(0.0, t - self.args.paper2_history_s)
        self.mode_fit_samples = [
            sample for sample in self.mode_fit_samples if sample[0] >= keep_after
        ]
        if t < self.args.paper2_after_s:
            return None

        window_start = max(0.0, t - self.args.paper2_window_s)
        window = [
            sample
            for sample in self.mode_fit_samples
            if window_start <= sample[0] <= t
        ]
        if len(window) < self.args.paper2_min_samples:
            return None

        times = [sample[0] for sample in window]
        values = [sample[1] for sample in window]
        mean_value = sum(values) / len(values)
        centered = [value - mean_value for value in values]
        p2p = max(centered) - min(centered)
        if p2p < self.args.paper2_min_p2p_mm:
            return None

        extrema: list[tuple[float, float, int]] = []
        radius = max(1, self.args.paper2_smooth_radius)
        smooth: list[float] = []
        for i in range(len(centered)):
            lo = max(0, i - radius)
            hi = min(len(centered), i + radius + 1)
            smooth.append(sum(centered[lo:hi]) / (hi - lo))

        for i in range(1, len(smooth) - 1):
            prev_v, value, next_v = smooth[i - 1], smooth[i], smooth[i + 1]
            sign = 0
            if value > prev_v and value >= next_v:
                sign = 1
            elif value < prev_v and value <= next_v:
                sign = -1
            if sign == 0 or abs(value) < self.args.paper2_min_peak_mm:
                continue
            if extrema and times[i] - extrema[-1][0] < self.args.paper2_min_peak_dt:
                if abs(value) > abs(extrema[-1][1]):
                    extrema[-1] = (times[i], value, sign)
                continue
            if extrema and extrema[-1][2] == sign:
                if abs(value) > abs(extrema[-1][1]):
                    extrema[-1] = (times[i], value, sign)
                continue
            extrema.append((times[i], value, sign))

        if len(extrema) < self.args.paper2_min_extrema:
            return None

        half_periods = [
            extrema[i][0] - extrema[i - 1][0]
            for i in range(1, len(extrema))
            if extrema[i][2] != extrema[i - 1][2]
        ]
        half_periods = [
            hp for hp in half_periods
            if self.args.zv_t_min <= hp <= self.args.zv_t_max
        ]
        if len(half_periods) < max(1, self.args.paper2_min_extrema - 1):
            return None
        half_periods.sort()
        mid = len(half_periods) // 2
        T = (
            half_periods[mid]
            if len(half_periods) % 2 == 1
            else 0.5 * (half_periods[mid - 1] + half_periods[mid])
        )
        omega_n = math.pi / T

        zeta = 0.0
        same_sign_pairs = []
        for i in range(2, len(extrema)):
            if extrema[i][2] == extrema[i - 2][2]:
                a0 = abs(extrema[i - 2][1])
                a1 = abs(extrema[i][1])
                if a0 > self.args.paper2_min_peak_mm and a1 > 1.0e-6 and a0 > a1:
                    same_sign_pairs.append((a0, a1))
        if same_sign_pairs:
            decrements = [math.log(a0 / a1) for a0, a1 in same_sign_pairs]
            delta = sum(decrements) / len(decrements)
            zeta = delta / math.sqrt(4.0 * math.pi * math.pi + delta * delta)
            zeta = max(self.args.id_zeta_min, min(zeta, self.args.id_zeta_max))

        est = Estimate(
            t=t,
            x=self.latest_swing_mm,
            i1=float('nan'),
            i2=float('nan'),
            i3=float('nan'),
            cond_b=1.0,
            omega_n=omega_n,
            zeta=zeta,
            T=T,
            A0=float('nan'),
            A1=float('nan'),
            A2=float('nan'),
        )
        self.latest_paper2_estimate = est
        if t - self._last_mode_fit_log_t >= self.args.print_period:
            self._last_mode_fit_log_t = t
            self.get_logger().info(
                f'paper2-step ID: t={t:.3f}s T={T:.4f}s omega={omega_n:.4f} '
                f'zeta={zeta:.4f} extrema={len(extrema)} p2p={p2p:.2f}mm '
                f'half_periods={len(half_periods)}')
        return est

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

    def _configure_model_schedule(self):
        omega_n = math.sqrt(self.args.gravity_m_s2 / self.args.robust_rope_length_m)
        T = self.args.robust_t_scale * math.pi / omega_n
        if self.args.profile == 'zv':
            A0, A1, A2 = self._standard_zv_amplitudes(self.args.robust_zeta)
            source = f'nonrobust_zv_L{self.args.robust_rope_length_m:.3f}m'
        else:
            A0, A1, A2 = self._standard_zvd_amplitudes(self.args.robust_zeta)
            source = f'robust_zvd_L{self.args.robust_rope_length_m:.3f}m'
        self.initial_velocity_mm_s = self.direction * A0 * self.vmax_abs_mm_s
        self.schedule = LockedSchedule(
            source=source,
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

    def _update_nonzero_ic_frequency_estimate(
        self,
    ) -> FreeSwingFrequencyEstimate:
        """Fit the latest tau seconds without blocking ROS sample callbacks."""
        now = time.monotonic()
        pending = self._nonzero_ic_fit_future
        if pending is not None:
            if not pending.done():
                raise NonzeroIcNotReady('adaptive frequency fit is running')
            self._nonzero_ic_fit_future = None
            try:
                estimate = pending.result()
            except Exception as exc:
                self._last_nonzero_ic_fit_error = str(exc)
                raise NonzeroIcNotReady(str(exc)) from exc
            return self._accept_nonzero_ic_frequency_estimate(estimate, now)

        if (
            now - self._last_nonzero_ic_fit_wall
            < self.args.nonzero_ic_fit_update_period_s
        ):
            if self.nonzero_ic_frequency_estimate is not None:
                return self.nonzero_ic_frequency_estimate
            raise NonzeroIcNotReady(self._last_nonzero_ic_fit_error)
        self._last_nonzero_ic_fit_wall = now
        self.nonzero_ic_frequency_estimate = None

        if not self.nonzero_ic_swing_samples:
            self._last_nonzero_ic_fit_error = 'no free-swing samples are available'
            raise NonzeroIcNotReady(self._last_nonzero_ic_fit_error)
        end_time_s = self.nonzero_ic_swing_samples[-1][0]
        window_start_s = end_time_s - self.args.tau
        window = [
            sample
            for sample in self.nonzero_ic_swing_samples
            if sample[0] >= window_start_s
        ]
        if len(window) < self.args.nonzero_ic_min_fit_samples:
            self._last_nonzero_ic_fit_error = (
                f'adaptive frequency fit has {len(window)}/'
                f'{self.args.nonzero_ic_min_fit_samples} required samples'
            )
            raise NonzeroIcNotReady(self._last_nonzero_ic_fit_error)
        window_duration_s = window[-1][0] - window[0][0]
        minimum_window_s = max(0.5, 0.90 * self.args.tau)
        if window_duration_s < minimum_window_s:
            self._last_nonzero_ic_fit_error = (
                f'adaptive frequency window is {window_duration_s:.2f}/'
                f'{minimum_window_s:.2f}s'
            )
            raise NonzeroIcNotReady(self._last_nonzero_ic_fit_error)

        try:
            executor = self._nonzero_ic_fit_executor
            if executor is not None:
                self._nonzero_ic_fit_future = executor.submit(
                    estimate_free_swing_frequency,
                    tuple(window),
                    omega_min_rad_s=self.args.nonzero_ic_omega_min_rad_s,
                    omega_max_rad_s=self.args.nonzero_ic_omega_max_rad_s,
                    grid_count=self.args.nonzero_ic_frequency_grid_count,
                )
                raise NonzeroIcNotReady('adaptive frequency fit is running')
            estimate = estimate_free_swing_frequency(
                window,
                omega_min_rad_s=self.args.nonzero_ic_omega_min_rad_s,
                omega_max_rad_s=self.args.nonzero_ic_omega_max_rad_s,
                grid_count=self.args.nonzero_ic_frequency_grid_count,
            )
        except ValueError as exc:
            self._last_nonzero_ic_fit_error = str(exc)
            raise NonzeroIcNotReady(str(exc)) from exc

        return self._accept_nonzero_ic_frequency_estimate(estimate, now)

    def _accept_nonzero_ic_frequency_estimate(
        self,
        estimate: FreeSwingFrequencyEstimate,
        now: float,
    ) -> FreeSwingFrequencyEstimate:
        """Apply quality gates and publish a completed worker estimate."""

        grid_step = (
            self.args.nonzero_ic_omega_max_rad_s
            - self.args.nonzero_ic_omega_min_rad_s
        ) / max(self.args.nonzero_ic_frequency_grid_count - 1, 1)
        if (
            estimate.omega_n_rad_s
            <= self.args.nonzero_ic_omega_min_rad_s + grid_step
            or estimate.omega_n_rad_s
            >= self.args.nonzero_ic_omega_max_rad_s - grid_step
        ):
            self._last_nonzero_ic_fit_error = (
                f'adaptive omega={estimate.omega_n_rad_s:.4f}rad/s is at the '
                'frequency-search boundary'
            )
            raise NonzeroIcNotReady(self._last_nonzero_ic_fit_error)
        if estimate.amplitude_mm < self.args.nonzero_ic_min_fit_amplitude_mm:
            self._last_nonzero_ic_fit_error = (
                f'free-swing amplitude={estimate.amplitude_mm:.2f}mm is below '
                f'{self.args.nonzero_ic_min_fit_amplitude_mm:.2f}mm'
            )
            raise NonzeroIcNotReady(self._last_nonzero_ic_fit_error)
        if estimate.normalized_rmse > self.args.nonzero_ic_max_fit_nrmse:
            self._last_nonzero_ic_fit_error = (
                f'free-swing fit NRMSE={estimate.normalized_rmse:.3f} exceeds '
                f'{self.args.nonzero_ic_max_fit_nrmse:.3f}'
            )
            raise NonzeroIcNotReady(self._last_nonzero_ic_fit_error)

        self.nonzero_ic_frequency_estimate = estimate
        self._last_nonzero_ic_fit_error = ''
        if now - self._last_nonzero_ic_fit_log_wall >= 1.0:
            self._last_nonzero_ic_fit_log_wall = now
            effective_length_m = (
                self.args.gravity_m_s2 / estimate.omega_n_rad_s**2
            )
            self.get_logger().info(
                'Adaptive free-swing ID: '
                f'tau={estimate.window_duration_s:.3f}s '
                f'n={estimate.sample_count} '
                f'omega_n={estimate.omega_n_rad_s:.6f}rad/s '
                f'L_eff={effective_length_m:.4f}m '
                f'amplitude={estimate.amplitude_mm:.2f}mm '
                f'RMSE={estimate.rmse_mm:.2f}mm '
                f'NRMSE={estimate.normalized_rmse:.4f}'
            )
        return estimate

    def _configure_nonzero_ic_schedule(
        self, state_capture_wall_override: float | None = None
    ) -> float:
        """Capture the fitted live state and solve the supplied closed form.

        Return the monotonic wall time represented by the captured state.  A
        frequency grid fit can take several control ticks, so the fitted state
        is projected from the latest payload sample to this time before the
        closed-form schedule is solved.
        """
        frequency_estimate: FreeSwingFrequencyEstimate | None = None
        fitted_swing_mm: float | None = None
        fitted_relative_velocity_mm_s: float | None = None
        fitted_command_start_swing_mm: float | None = None
        fitted_command_start_relative_velocity_mm_s: float | None = None
        state_capture_wall = time.monotonic()
        state_prediction_age_s = 0.0
        if self.args.nonzero_ic_adaptive_frequency:
            if (
                self.latest_payload_time is None
                or self.latest_payload_wall is None
                or self.payload_clock_offset_s is None
            ):
                raise NonzeroIcNotReady(
                    'calibrated payload sample time is unavailable')
            frequency_estimate = self._update_nonzero_ic_frequency_estimate()
            # The grid fit above is intentionally performed before selecting
            # the command-start state.  Project the fitted sinusoid across the
            # elapsed wall time so solver latency does not make x0/v0 stale.
            state_capture_wall = (
                time.monotonic()
                if state_capture_wall_override is None
                else float(state_capture_wall_override)
            )
            projected_payload_time = payload_time_at_wall(
                state_capture_wall,
                self.payload_clock_offset_s,
            )
            state_prediction_age_s = max(
                0.0,
                projected_payload_time - self.latest_payload_time,
            )
            fitted_swing_mm, fitted_relative_velocity_mm_s = (
                frequency_estimate.oscillation_state_at(projected_payload_time)
            )
            command_start_wall = state_capture_wall - (
                1.0e-3 * self.args.timed_profile_actuation_lead_ms
                if self.args.controller_timed_profile
                else 0.0
            )
            command_start_payload_time = payload_time_at_wall(
                command_start_wall,
                self.payload_clock_offset_s,
            )
            (
                fitted_command_start_swing_mm,
                fitted_command_start_relative_velocity_mm_s,
            ) = frequency_estimate.oscillation_state_at(
                command_start_payload_time)

        if self.args.initial_swing_mm is not None:
            world_swing_mm = float(self.args.initial_swing_mm)
        elif fitted_swing_mm is not None:
            world_swing_mm = fitted_swing_mm
        elif self.latest_swing_mm is not None and math.isfinite(self.latest_swing_mm):
            world_swing_mm = float(self.latest_swing_mm)
        else:
            raise ValueError(
                'initial swing is unavailable; provide --initial-swing-mm or '
                'a fresh payload relative-position sample'
            )

        if self.args.initial_payload_velocity_mm_s is not None:
            world_payload_velocity_mm_s = float(
                self.args.initial_payload_velocity_mm_s
            )
        else:
            relative_velocity_mm_s = (
                fitted_relative_velocity_mm_s
                if fitted_relative_velocity_mm_s is not None
                else (
                    self.latest_payload_rel_vx_mm_s
                    if self.args.axis == 'x'
                    else self.latest_payload_rel_vy_mm_s
                )
            )
            cart_sample = self.latest_cart_state_sample or self.latest_gantry_state_sample
            if relative_velocity_mm_s is None or cart_sample is None:
                raise ValueError(
                    'initial payload velocity is unavailable; provide '
                    '--initial-payload-velocity-mm-s or publish velocity in '
                    '/payload/pose_e_rel'
                )
            cart_velocity_mm_s = 1000.0 * float(
                cart_sample.vx if self.args.axis == 'x' else cart_sample.vy
            )
            world_payload_velocity_mm_s = (
                float(relative_velocity_mm_s) + cart_velocity_mm_s
            )

        if not math.isfinite(world_payload_velocity_mm_s):
            raise ValueError('initial payload velocity must be finite')

        # The closed form is written in a positive direction-of-travel frame.
        # Transform world-axis measurements for negative moves, then transform
        # the resulting positive velocity command back in _axis_velocity_for_time.
        signed_swing_mm = self.direction * world_swing_mm
        signed_payload_velocity_mm_s = (
            self.direction * world_payload_velocity_mm_s
        )
        if self.args.nonzero_ic_start_at_peak:
            velocity_tolerance = (
                self.args.nonzero_ic_peak_velocity_tolerance_mm_s
            )
            if signed_swing_mm <= 0.0:
                raise NonzeroIcNotReady(
                    'waiting for the positive payload peak in the direction '
                    f'of travel: signed swing={signed_swing_mm:.2f}mm'
                )
            if abs(signed_payload_velocity_mm_s) > velocity_tolerance:
                raise NonzeroIcNotReady(
                    'waiting for zero payload velocity at the positive peak: '
                    f'payload_v={world_payload_velocity_mm_s:.2f}mm/s, '
                    f'tolerance=+/-{velocity_tolerance:.2f}mm/s'
                )
        if frequency_estimate is not None:
            fitted_omega_n = frequency_estimate.omega_n_rad_s
        else:
            fitted_omega_n = (
                self.args.nonzero_ic_omega_rad_s
                if self.args.nonzero_ic_omega_rad_s > 0.0
                else math.sqrt(
                    self.args.gravity_m_s2 / self.args.robust_rope_length_m
                )
            )
        solver_fit_omega_n = fitted_omega_n
        amplitude_correction = None
        if (
            frequency_estimate is not None
            and self.args.nonzero_ic_finite_amplitude_correction
        ):
            try:
                amplitude_correction = correct_finite_amplitude_frequency(
                    fitted_omega_n,
                    frequency_estimate.amplitude_mm,
                    self.args.robust_rope_length_m,
                )
            except ValueError as exc:
                raise NonzeroIcNotReady(
                    f'finite-amplitude frequency correction failed: {exc}'
                ) from exc
            solver_fit_omega_n = amplitude_correction.small_angle_omega_rad_s
        omega_n = select_nonzero_ic_shaper_frequency(
            solver_fit_omega_n,
            self.args.nonzero_ic_shaper_frequency_scale,
            self.args.nonzero_ic_shaper_omega_rad_s,
        )
        shaper_frequency_mode = (
            'fixed'
            if self.args.nonzero_ic_shaper_omega_rad_s > 0.0
            else (
                'amplitude-corrected-scaled-fit'
                if amplitude_correction is not None
                else 'scaled-fit'
            )
        )
        robust_solution: RobustNonzeroIcShaper | None = None
        try:
            if self.args.nonzero_ic_robust:
                robust_solution = solve_robust_nonzero_ic_shaper(
                    initial_swing_mm=signed_swing_mm,
                    initial_payload_velocity_mm_s=signed_payload_velocity_mm_s,
                    maximum_speed_mm_s=self.vmax_abs_mm_s,
                    omega_n_rad_s=omega_n,
                    move_duration_s=self.tf,
                    frequency_band_fraction=(
                        self.args.nonzero_ic_robust_band_fraction
                    ),
                )
                solution = None
                positive_profile_events = tuple(
                    (time_s, gain * self.vmax_abs_mm_s)
                    for time_s, gain in robust_solution.gain_events()
                )
                schedule_tail_s = robust_solution.tail_s
                schedule_amplitudes = robust_solution.start_amplitudes
                terminal_position_residual_mm = (
                    robust_solution.terminal_position_residual_mm
                )
                terminal_velocity_residual_mm_s = (
                    robust_solution.terminal_velocity_residual_mm_s
                )
            else:
                solution = solve_nonzero_ic_shaper(
                    initial_swing_mm=signed_swing_mm,
                    initial_payload_velocity_mm_s=signed_payload_velocity_mm_s,
                    maximum_speed_mm_s=self.vmax_abs_mm_s,
                    omega_n_rad_s=omega_n,
                    move_duration_s=self.tf,
                    maximum_absolute_gain=(
                        self.args.nonzero_ic_max_command_speed_mm_s
                        / self.vmax_abs_mm_s
                    ),
                )
                if not solution.is_forward_only:
                    raise NonzeroIcNotReady(
                        f'forward-only gains not ready: A0={solution.A0:.6f}, '
                        f'A1={solution.A1:.6f}'
                    )
                positive_profile_events = (
                    (0.0, solution.A0 * self.vmax_abs_mm_s),
                    (solution.switch_time_s, self.vmax_abs_mm_s),
                    (self.tf, solution.A1 * self.vmax_abs_mm_s),
                    (self.tf + solution.switch_time_s, 0.0),
                )
                schedule_tail_s = solution.switch_time_s
                schedule_amplitudes = (solution.A0, solution.A1, 0.0)
                terminal_position_residual_mm = (
                    solution.terminal_position_residual_mm
                )
                terminal_velocity_residual_mm_s = (
                    solution.terminal_velocity_residual_mm_s
                )
        except ValueError as exc:
            raise NonzeroIcNotReady(str(exc)) from exc
        execution_cart_sample = (
            self.latest_cart_state_sample or self.latest_gantry_state_sample
        )
        if execution_cart_sample is None:
            raise ValueError('cart position is unavailable for workspace validation')
        start_cart_mm = 1000.0 * float(
            execution_cart_sample.x
            if self.args.axis == 'x'
            else execution_cart_sample.y
        )
        signed_event_positions = [0.0]
        position_mm = 0.0
        previous_time_s = 0.0
        previous_velocity_mm_s = 0.0
        for event_time_s, velocity_mm_s in positive_profile_events:
            position_mm += previous_velocity_mm_s * (
                event_time_s - previous_time_s
            )
            signed_event_positions.append(position_mm)
            previous_time_s = event_time_s
            previous_velocity_mm_s = velocity_mm_s
        signed_event_positions_mm = tuple(signed_event_positions)
        if abs(position_mm - self.args.target_distance_mm) > 1.0e-5:
            raise ValueError(
                'nonzero-IC profile failed exact-travel verification: '
                f'predicted={position_mm:.6f}mm, '
                f'target={self.args.target_distance_mm:.6f}mm')
        world_event_positions_mm = tuple(
            start_cart_mm + self.direction * position
            for position in signed_event_positions_mm
        )
        workspace_low_mm = (
            self.args.workspace_min_mm + self.args.workspace_margin_mm
        )
        workspace_high_mm = (
            self.args.workspace_max_mm - self.args.workspace_margin_mm
        )
        if (
            min(world_event_positions_mm) < workspace_low_mm
            or max(world_event_positions_mm) > workspace_high_mm
        ):
            raise ValueError(
                'nonzero-IC transient leaves the configured workspace: '
                f'predicted=[{min(world_event_positions_mm):.1f}, '
                f'{max(world_event_positions_mm):.1f}]mm, '
                f'allowed=[{workspace_low_mm:.1f}, {workspace_high_mm:.1f}]mm'
            )
        self.nonzero_ic_initial_swing_mm = world_swing_mm
        self.nonzero_ic_initial_payload_velocity_mm_s = (
            world_payload_velocity_mm_s
        )
        self.nonzero_ic_predicted_peak_swing_mm = world_swing_mm
        self.nonzero_ic_predicted_peak_payload_velocity_mm_s = (
            world_payload_velocity_mm_s
        )
        if fitted_command_start_swing_mm is not None:
            self.nonzero_ic_predicted_command_start_swing_mm = (
                fitted_command_start_swing_mm
            )
        if fitted_command_start_relative_velocity_mm_s is not None:
            command_cart_sample = (
                self.latest_cart_state_sample or self.latest_gantry_state_sample
            )
            command_cart_velocity_mm_s = (
                0.0
                if command_cart_sample is None
                else 1000.0 * float(
                    command_cart_sample.vx
                    if self.args.axis == 'x'
                    else command_cart_sample.vy
                )
            )
            self.nonzero_ic_predicted_command_start_payload_velocity_mm_s = (
                fitted_command_start_relative_velocity_mm_s
                + command_cart_velocity_mm_s
            )
        self.nonzero_ic_fitted_omega_rad_s = fitted_omega_n
        self.nonzero_ic_small_angle_omega_rad_s = solver_fit_omega_n
        self.nonzero_ic_fitted_amplitude_angle_deg = (
            None
            if amplitude_correction is None
            else math.degrees(amplitude_correction.amplitude_angle_rad)
        )
        self.nonzero_ic_frequency_correction_factor = (
            None
            if amplitude_correction is None
            else amplitude_correction.correction_factor
        )
        self.nonzero_ic_used_omega_rad_s = omega_n
        self.nonzero_ic_robust_solution = robust_solution
        self.nonzero_ic_profile_events = positive_profile_events
        self.initial_velocity_mm_s = (
            self.direction * positive_profile_events[0][1]
        )
        source_base = (
            (
                'nonzero_ic_adaptive_free_swing_amplitude_corrected'
                if amplitude_correction is not None
                else 'nonzero_ic_adaptive_free_swing'
            )
            if frequency_estimate is not None
            else 'nonzero_ic_closed_form'
        )
        if robust_solution is not None:
            source_base += (
                f'_robust_band_{robust_solution.frequency_band_fraction:.4f}'
            )
        self.schedule = LockedSchedule(
            source=source_base,
            locked_at=0.0,
            id_time=(
                float('nan')
                if frequency_estimate is None
                else frequency_estimate.window_duration_s
            ),
            id_T=(
                float('nan')
                if frequency_estimate is None
                else math.pi / frequency_estimate.omega_n_rad_s
            ),
            T=schedule_tail_s,
            A0=schedule_amplitudes[0],
            A1=schedule_amplitudes[1],
            A2=schedule_amplitudes[2],
            raw_id_zeta=0.0,
            id_zeta=0.0,
            shaper_zeta=0.0,
            cond_b=float('nan'),
        )
        signed_profile_events = tuple(
            (event_time_s, self.direction * velocity_mm_s)
            for event_time_s, velocity_mm_s in positive_profile_events
        )
        switches = tuple(event[0] for event in signed_profile_events[1:])
        levels = tuple(event[1] for event in signed_profile_events)
        correction_log = (
            ''
            if amplitude_correction is None
            else (
                f'fit_amplitude_angle='
                f'{math.degrees(amplitude_correction.amplitude_angle_rad):.3f}deg, '
                f'omega_small_angle='
                f'{amplitude_correction.small_angle_omega_rad_s:.6f}rad/s, '
                f'nonlinear_factor='
                f'{amplitude_correction.correction_factor:.8f}, '
            )
        )
        robust_log = (
            ''
            if robust_solution is None
            else (
                f'robust_band=+/-{100.0 * robust_solution.frequency_band_fraction:.1f}%, '
                f'start_weights={robust_solution.start_amplitudes}, '
                f'stop_weights={robust_solution.stop_amplitudes}, '
                f'impulse_times={robust_solution.impulse_times_s}s, '
                f'band_worst={robust_solution.worst_case_residual_fraction:.4f}, '
                f'two_impulse_band_worst='
                f'{robust_solution.baseline_worst_case_residual_fraction:.4f}, '
            )
        )
        self.get_logger().info(
            'LOCKED robust nonzero-IC shaper at forward-only profile start: '
            if robust_solution is not None else
            'LOCKED nonzero-IC shaper at forward-only profile start: '
        )
        self.get_logger().info(
            f'x0=[swing={world_swing_mm:.3f}mm, '
            f'payload_v={world_payload_velocity_mm_s:.3f}mm/s] world-axis, '
            f'omega_finite_fit={fitted_omega_n:.6f}rad/s, '
            + correction_log
            + f'omega_shaper={omega_n:.6f}rad/s '
            f'(mode={shaper_frequency_mode}, '
            f'scale={self.args.nonzero_ic_shaper_frequency_scale:.6f}), '
            + robust_log
            + f'tail={schedule_tail_s:.6f}s, '
            f'state_prediction={1000.0 * state_prediction_age_s:.1f}ms'
        )
        if self.nonzero_ic_predicted_command_start_payload_velocity_mm_s is not None:
            self.get_logger().info(
                'Actuator-delay model states: '
                f'command issued {self.args.timed_profile_actuation_lead_ms:.1f}ms '
                'before the intended physical peak; '
                f'predicted command-start payload_v='
                f'{self.nonzero_ic_predicted_command_start_payload_velocity_mm_s:.3f}'
                'mm/s, predicted physical-peak payload_v='
                f'{world_payload_velocity_mm_s:.3f}mm/s'
            )
        self.get_logger().info(
            'Nonzero-IC switches: '
            + ', '.join(f'{value:.6f}s' for value in switches)
            + ' | velocities: '
            + ', '.join(f'{value:.3f}' for value in levels)
            + ' mm/s | verified terminal residual: '
            f'{terminal_position_residual_mm:.3e}mm, '
            f'{terminal_velocity_residual_mm_s:.3e}mm/s'
        )
        return state_capture_wall

    def _excite_axis_angle_deg(self) -> float | None:
        return (
            self.latest_enc_pitch_deg
            if self.args.axis == 'x'
            else self.latest_enc_roll_deg
        )

    def _excite_log_values(self) -> list:
        """Nine trailing CSV fields describing the swing-up stage (blank-safe)."""
        raw_angle = self._excite_axis_angle_deg()
        cart_offset = (
            ''
            if self.latest_cart_q_mm is None or self.excite_start_cart_q_mm is None
            else self.latest_cart_q_mm - self.excite_start_cart_q_mm
        )
        omega = '' if self.exciter is None else self.exciter.cfg.omega_rad_s
        cmd = self._latest_excite_cmd
        return [
            '' if raw_angle is None else raw_angle,
            self.excite_angle_bias_deg,
            '' if cmd is None else cmd.amplitude_est_deg,
            '' if cmd is None else cmd.peak_est_deg,
            '' if cmd is None else cmd.angle_rate_deg_s,
            '' if cmd is None else cmd.velocity_mm_s,
            '' if self.exciter is None else self.exciter.drive_sign,
            cart_offset,
            omega,
        ]

    def _cart_within_workspace(self) -> bool:
        cart_m = self._current_cart_axis_m(self.latest_gantry_state_sample)
        if cart_m is None:
            cart_m = self._current_cart_axis_m()
        if cart_m is None:
            return True
        cart_mm = cart_m * 1000.0
        low = self.args.workspace_min_mm + self.args.workspace_margin_mm
        high = self.args.workspace_max_mm - self.args.workspace_margin_mm
        return low <= cart_mm <= high

    def _runtime_safety_reason(self, now: float) -> str | None:
        """Return a reason to stop a live nonzero-IC run, or ``None``."""
        state = self.latest_gantry_state_sample
        if (
            state is None
            or now - state.rx_wall > self.args.gantry_fresh_timeout
        ):
            age_ms = (
                float('inf') if state is None
                else 1000.0 * max(0.0, now - state.rx_wall)
            )
            return (
                'gantry state is stale during nonzero-IC execution: '
                f'age={age_ms:.1f}ms, limit='
                f'{1000.0 * self.args.gantry_fresh_timeout:.1f}ms'
            )
        if (
            self.args.require_encoder_health
            and not self._encoder_health_ready(now)
        ):
            return 'encoder diagnostics are missing, stale, or unhealthy'

        cart_mm = 1000.0 * float(
            state.x if self.args.axis == 'x' else state.y
        )
        low = self.args.workspace_min_mm + self.args.workspace_margin_mm
        high = self.args.workspace_max_mm - self.args.workspace_margin_mm
        if not (low <= cart_mm <= high):
            return (
                f'cart left the configured workspace margin: position='
                f'{cart_mm:.1f}mm allowed=[{low:.1f}, {high:.1f}]mm'
            )

        if self.phase == 'maneuver' and self.start_cart_q_mm is not None:
            traveled_mm = self.direction * (cart_mm - self.start_cart_q_mm)
            if traveled_mm > self.args.max_travel_mm:
                return (
                    f'actual travel {traveled_mm:.1f}mm exceeded '
                    f'--max-travel-mm={self.args.max_travel_mm:.1f}mm'
                )
        return None

    def _abort_excite(self, now: float, reason: str) -> None:
        # Anchor wall_start so the main loop's abort handling collects residual
        # and finishes the run instead of looping in the excite branch.
        self.wall_start = now
        self.phase = 'residual'
        self._abort_without_estimate(0.0, reason=reason)

    def _run_excite_phase(self, now: float) -> None:
        """One control tick of the closed-loop swing-up stage."""
        angle_raw = self._excite_axis_angle_deg()
        enc_fresh = (
            self.latest_enc_wall is not None
            and now - self.latest_enc_wall <= self.args.payload_fresh_timeout
        )
        if angle_raw is None or not enc_fresh or not math.isfinite(angle_raw):
            self._publish_stream(0.0, 0.0)
            if now - self._last_excite_wait_log_wall >= 1.0:
                self._last_excite_wait_log_wall = now
                self.get_logger().warn(
                    'Excite: holding zero; waiting for a fresh '
                    f'/payload/pose_e {self.args.axis} angle')
            return

        if not self._cart_within_workspace():
            self._abort_excite(
                now, reason='cart left the configured workspace during excitation')
            return

        cart_offset_mm = (
            0.0
            if self.latest_cart_q_mm is None or self.excite_start_cart_q_mm is None
            else self.latest_cart_q_mm - self.excite_start_cart_q_mm
        )
        if abs(cart_offset_mm) > 1.5 * self.args.excite_travel_budget_mm:
            self._abort_excite(
                now,
                reason=(
                    f'excitation cart offset {cart_offset_mm:.0f}mm exceeded '
                    f'1.5x the travel budget'))
            return

        angle_deg = angle_raw - self.excite_angle_bias_deg
        cmd = self.exciter.update(now, angle_deg, cart_offset_mm)
        self._latest_excite_cmd = cmd

        if cmd.abort_reason is not None:
            self._abort_excite(now, reason=f'excitation aborted: {cmd.abort_reason}')
            return

        if cmd.converged:
            if getattr(self.args, 'excite_only', False):
                self.excite_only_complete = True
                self.wall_start = now
                self.final_zero_wall = now
                self.phase = 'residual'
                self._publish_stream(0.0, 0.0)
                self._log_row(0.0, 0.0, 0.0)
                self.get_logger().info(
                    'Bounded excitation-only trial returned to its anchor; '
                    f'collecting residual swing for {self.args.residual_window:.1f}s. '
                    'The nonzero-IC travel maneuver will not run.')
                return
            self.phase = 'id_hold'
            self.id_hold_start_wall = now
            self._publish_stream(0.0, 0.0)
            self._log_row(-(now - self.run_start_wall), 0.0, 0.0)
            self.get_logger().info(
                f'Bounded swing-up converged and returned to its anchor: '
                f'|theta|~{cmd.amplitude_est_deg:.2f}deg '
                f'(target {self.args.excite_target_angle_deg:.1f}deg, '
                f'cart offset={cart_offset_mm:+.2f}mm). Holding zero for '
                f'{self.args.tau:.1f}s of stationary-trolley ID.')
            return

        vx, vy = (
            (cmd.velocity_mm_s, 0.0)
            if self.args.axis == 'x'
            else (0.0, cmd.velocity_mm_s)
        )
        self._publish_stream(vx, vy)
        self._log_row(-(now - self.run_start_wall), vx, vy)

    def _next_positive_peak_wall(
        self,
        estimate: FreeSwingFrequencyEstimate,
        earliest_wall: float,
    ) -> float:
        """Map the first fitted positive peak after ``earliest_wall`` to wall time."""
        if self.payload_clock_offset_s is None:
            raise NonzeroIcNotReady(
                'payload clock is not calibrated for timed start')
        omega = estimate.omega_n_rad_s
        period = 2.0 * math.pi / omega
        positive_peak_phase = math.atan2(
            estimate.sine_coefficient_mm,
            estimate.cosine_coefficient_mm,
        )
        # The shaper's "positive" initial swing is positive in the direction
        # of travel, not necessarily positive in world coordinates.  A
        # negative-axis maneuver therefore starts at the fitted negative
        # world-axis peak, one half-period after the positive-world peak.
        if self.direction < 0.0:
            positive_peak_phase += math.pi
        base_peak_payload_time = (
            estimate.reference_time_s + positive_peak_phase / omega
        )
        earliest_payload_time = payload_time_at_wall(
            earliest_wall,
            self.payload_clock_offset_s,
        )
        cycles = math.ceil(
            (earliest_payload_time - base_peak_payload_time) / period
            - 1.0e-12
        )
        peak_payload_time = base_peak_payload_time + max(0, cycles) * period
        while peak_payload_time < earliest_payload_time - 1.0e-9:
            peak_payload_time += period
        return payload_wall_at_time(
            peak_payload_time,
            self.payload_clock_offset_s,
        )

    def _arm_controller_timed_profile(
        self,
        start_wall: float,
        fitted_peak_wall: float,
    ) -> None:
        if self.schedule is None:
            raise ValueError('cannot arm controller timing without a locked schedule')
        if not self.timed_profile_cli.service_is_ready():
            raise NonzeroIcNotReady('/gantry/execute_timed_profile is not available')

        schedule = self.schedule
        if self.nonzero_ic_profile_events is not None:
            raw_events = tuple(
                (event_time_s, self.direction * velocity_mm_s)
                for event_time_s, velocity_mm_s
                in self.nonzero_ic_profile_events
            )
        else:
            levels = (
                self.direction * schedule.A0 * self.vmax_abs_mm_s,
                self.direction * self.vmax_abs_mm_s,
                self.direction * schedule.A1 * self.vmax_abs_mm_s,
                0.0,
            )
            raw_events = (
                (0.0, levels[0]),
                (schedule.T, levels[1]),
                (self.tf, levels[2]),
                (self.tf + schedule.T, levels[3]),
            )
        events: list[tuple[float, float]] = []
        for event_time, velocity in raw_events:
            if events and math.isclose(
                event_time, events[-1][0], rel_tol=0.0, abs_tol=1.0e-10
            ):
                events[-1] = (event_time, velocity)
            else:
                events.append((event_time, velocity))

        lead_s = start_wall - time.monotonic()
        if lead_s < 0.10:
            raise NonzeroIcNotReady(
                f'controller-timed start lead shrank to {lead_s:.3f}s'
            )
        start_ros_ns = self.get_clock().now().nanoseconds + int(lead_s * 1.0e9)
        request = ExecuteTimedProfile.Request()
        request.start_time.sec = int(start_ros_ns // 1_000_000_000)
        request.start_time.nanosec = int(start_ros_ns % 1_000_000_000)
        request.time_s = [float(event[0]) for event in events]
        if self.args.axis == 'x':
            request.vx_mm_s = [float(event[1]) for event in events]
            request.vy_mm_s = [0.0] * len(events)
        else:
            request.vx_mm_s = [0.0] * len(events)
            request.vy_mm_s = [float(event[1]) for event in events]
        self.timed_profile_start_wall = start_wall
        self.timed_profile_peak_wall = fitted_peak_wall
        self.timed_profile_payload_clock_offset_s = self.payload_clock_offset_s
        self.timed_profile_arm_payload_queue_delay_ms = (
            self.latest_payload_queue_delay_ms
        )
        if self.timed_profile_payload_clock_offset_s is not None:
            self.timed_profile_peak_payload_time = payload_time_at_wall(
                fitted_peak_wall,
                self.timed_profile_payload_clock_offset_s,
            )
            self.timed_profile_start_payload_time = payload_time_at_wall(
                start_wall,
                self.timed_profile_payload_clock_offset_s,
            )
        else:
            self.timed_profile_peak_payload_time = None
            self.timed_profile_start_payload_time = None
        self.nonzero_ic_measured_command_start = None
        self.nonzero_ic_measured_peak = None
        self.timed_profile_request_wall = time.monotonic()
        self.timed_profile_future = self.timed_profile_cli.call_async(request)
        self.phase = 'arm_profile'

    def _poll_controller_timed_profile(self) -> bool:
        future = self.timed_profile_future
        if future is None:
            return False
        if not future.done():
            start_wall = self.timed_profile_start_wall
            if (
                start_wall is not None
                and time.monotonic() >= start_wall - 0.05
            ):
                raise ValueError(
                    'controller did not acknowledge the timed profile before '
                    'its scheduled start')
            return False
        self.timed_profile_future = None
        try:
            result = future.result()
        except Exception as exc:
            raise ValueError(f'controller-timed profile service failed: {exc}') from exc
        if result is None or not result.success:
            message = 'no response' if result is None else result.message
            raise ValueError(f'controller rejected timed profile: {message}')
        if self.timed_profile_start_wall is None:
            raise ValueError('controller accepted a timed profile without a start epoch')
        self.wall_start = self.timed_profile_start_wall
        self.start_cart_q_mm = self.latest_cart_q_mm
        self.final_zero_wall = None
        self.residual_samples.clear()
        self.controller_timed_profile_active = True
        self.phase = 'armed_profile'
        self.get_logger().info(
            'Controller accepted the nonzero-IC schedule; holding zero until '
            'the actuator-compensated profile start epoch')
        return True

    def _maybe_begin_nonzero_ic_profile(self, now: float) -> bool:
        """Start at the first live state with a forward-only valid solution."""
        if self.args.excite and self.phase != 'wait_peak':
            return False
        if (
            self.latest_payload_wall is None
            or now - self.latest_payload_wall > self.args.payload_fresh_timeout
        ):
            reason = 'payload state is not fresh'
        elif (
            getattr(self.args, 'controller_timed_profile', False)
            and (
                self.latest_payload_queue_delay_ms is None
                or self.latest_payload_queue_delay_ms
                > self.args.nonzero_ic_max_payload_queue_delay_ms
            )
        ):
            delay_text = (
                'unavailable'
                if self.latest_payload_queue_delay_ms is None
                else f'{self.latest_payload_queue_delay_ms:.2f}ms'
            )
            reason = (
                'payload sample waited in the callback queue: '
                f'delay={delay_text}, maximum='
                f'{self.args.nonzero_ic_max_payload_queue_delay_ms:.2f}ms'
            )
        else:
            try:
                if getattr(self.args, 'controller_timed_profile', False):
                    estimate = self._update_nonzero_ic_frequency_estimate()
                    actuation_lead_s = (
                        1.0e-3 * self.args.timed_profile_actuation_lead_ms
                    )
                    fitted_peak_wall = self._next_positive_peak_wall(
                        estimate,
                        now + self.args.timed_profile_lead_s + actuation_lead_s,
                    )
                    self._configure_nonzero_ic_schedule(
                        state_capture_wall_override=fitted_peak_wall)
                    profile_start_wall = actuator_compensated_profile_start(
                        fitted_peak_wall,
                        self.args.timed_profile_actuation_lead_ms,
                    )
                    self._arm_controller_timed_profile(
                        profile_start_wall,
                        fitted_peak_wall,
                    )
                    self.get_logger().info(
                        'Locked nonzero-IC schedule for controller-owned '
                        f'profile start in {profile_start_wall - now:.3f}s, '
                        f'{self.args.timed_profile_actuation_lead_ms:.1f}ms '
                        'before the fitted positive payload peak; '
                        f'payload queue delay='
                        f'{self.latest_payload_queue_delay_ms:.2f}ms')
                    return False
                state_capture_wall = self._configure_nonzero_ic_schedule()
            except NonzeroIcNotReady as exc:
                reason = str(exc)
            except ValueError as exc:
                self.wall_start = now
                self._abort_without_estimate(
                    0.0,
                    reason=f'Cannot execute nonzero-IC profile safely: {exc}',
                )
                return False
            else:
                # The schedule represents the projected payload state at
                # state_capture_wall, not the timer callback's stale `now`
                # captured before the frequency fit.  Starting from the stale
                # timestamp shortened the first velocity interval by the fit
                # latency (85 ms / 30 mm in the 2026-08-27 run).
                self.wall_start = state_capture_wall
                self.start_cart_q_mm = self.latest_cart_q_mm
                self.final_zero_wall = None
                self.residual_samples.clear()
                self.get_logger().info(
                    'Forward-only initial condition is ready; beginning the '
                    'nonzero-IC velocity profile now')
                return True

        if now - self._last_nonzero_ic_wait_log_wall >= 1.0:
            self._last_nonzero_ic_wait_reason = reason
            self._last_nonzero_ic_wait_log_wall = now
            self.get_logger().info(
                f'Holding zero; waiting for forward-only initial condition: {reason}')
        return False

    def _standard_zv_amplitudes(self, zeta: float) -> tuple[float, float, float]:
        z = max(0.0, min(float(zeta), 0.99))
        wd_factor = math.sqrt(max(1.0 - z * z, 1.0e-12))
        decay = math.exp(-math.pi * z / wd_factor)
        denom = 1.0 + decay
        return 1.0 / denom, decay / denom, 0.0

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

        coeffs = np.array(
            [
                -K * sin_phi,
                K - 1.0 + 3.0 * K * cos_phi,
                3.0 * K * sin_phi,
                K - K * cos_phi - 1.0,
            ],
            dtype=float,
        )
        coeffs = np.trim_zeros(coeffs, trim='f')
        if len(coeffs) <= 1:
            return None
        try:
            roots = np.roots(coeffs)
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

    def _colleague_schedule_feasible_for_estimate(self, est: Estimate) -> bool:
        if self.args.profile != 'colleague-paper-closed':
            return True
        solved = self._colleague_paper_closed_form_amplitudes(est.omega_n)
        if solved is None:
            return False
        _A1, schedule_T = solved
        if not (self.args.zv_t_min <= schedule_T <= self.args.zv_t_max):
            return False
        if self.tf <= self.args.tau + 2.0 * schedule_T:
            return False
        if (
            self.args.is2_schedule_margin_s > 0.0
            and self.tf - (self.args.tau + 2.0 * schedule_T) < self.args.is2_schedule_margin_s
        ):
            return False
        return True

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
        if self._last_stream_wall is not None:
            self.latest_stream_dt_ms = 1000.0 * max(0.0, now - self._last_stream_wall)
        self._last_stream_wall = now
        if (
            self.args.profile == 'nonzero-ic'
            and self.motion_started
            and not self.aborted
            and self.phase in (
                'excite', 'id_hold', 'wait_peak', 'arm_profile',
                'armed_profile', 'maneuver')
        ):
            safety_reason = self._runtime_safety_reason(now)
            if safety_reason is not None:
                if self.wall_start is None:
                    self.wall_start = now
                self.phase = 'residual'
                self._abort_without_estimate(
                    0.0,
                    reason=f'Runtime safety stop: {safety_reason}',
                )
                return
        if self.wall_start is None:
            if not (self.args.profile == 'nonzero-ic' and self.motion_started):
                self._publish_stream(0.0, 0.0)
                return

            if self.phase == 'excite':
                self._run_excite_phase(now)
                return

            if self.phase == 'id_hold':
                self._publish_stream(0.0, 0.0)
                self._log_row(-(now - self.run_start_wall), 0.0, 0.0)
                if now - self.id_hold_start_wall >= self.args.tau:
                    self.phase = 'wait_peak'
                    self.get_logger().info(
                        f'Free-swing ID hold complete ({self.args.tau:.1f}s); '
                        'waiting for a zero-velocity swing peak to start the '
                        'shaped maneuver')
                return

            if self.phase == 'arm_profile':
                self._publish_stream(0.0, 0.0)
                self._log_row(-(now - self.run_start_wall), 0.0, 0.0)
                try:
                    self._poll_controller_timed_profile()
                except ValueError as exc:
                    self.wall_start = now
                    self.phase = 'residual'
                    self._abort_without_estimate(
                        0.0, reason=f'Cannot arm controller timing safely: {exc}')
                return

            # phase == 'wait_peak' (also the entry phase without --excite)
            self._publish_stream(0.0, 0.0)
            # Keep the CSV continuous while the trolley is stationary and the
            # payload phase is being monitored.  Previously this branch
            # returned without logging, so plots drew a false straight line
            # across the missing wait-for-peak interval.
            self._log_row(-(now - self.run_start_wall), 0.0, 0.0)
            if not self._maybe_begin_nonzero_ic_profile(now):
                return
            self.phase = 'maneuver'
            # Begin the first nonzero interval in this callback instead of
            # losing another stream period before the initial command.
            now = time.monotonic()

        controller_timed_active = getattr(
            self, 'controller_timed_profile_active', False)
        if (
            controller_timed_active
            and self.phase == 'armed_profile'
            and now < self.wall_start
        ):
            self._log_row(now - self.wall_start, 0.0, 0.0)
            return
        if self.phase == 'armed_profile':
            self.phase = 'maneuver'

        move_t = now - self.wall_start
        if getattr(self, 'excite_only_complete', False):
            self._publish_stream(0.0, 0.0)
            self._log_row(move_t, 0.0, 0.0)
            if now - self.final_zero_wall >= self.args.residual_window:
                self._finish_run()
            return
        if self.aborted:
            if not controller_timed_active:
                self._publish_stream(0.0, 0.0)
            # Preserve the requested residual window for aborted runs too.
            # Previously the node collected residual statistics but stopped
            # writing CSV rows at the abort instant.
            self._log_row(move_t, 0.0, 0.0)
            if now - self.final_zero_wall >= self.args.residual_window:
                self._finish_run()
            return

        if (
            self._is_colleague_profile()
            and self.schedule is None
            and move_t >= self.args.tau
        ):
            source, selected = self._fixed_id_estimate(move_t)
            if selected is None:
                source, selected = self._select_colleague_lock_estimate(move_t)
            latest = self._latest_active_estimate()
            if selected is not None:
                self.get_logger().info(
                    f'Using {source} estimate at tau: '
                    f'id_t={selected.t:.3f}s T={selected.T:.4f}s omega={selected.omega_n:.4f} '
                    f'zeta={selected.zeta:.4f} '
                    f'condB={selected.cond_b:.3e}')
                self._lock_colleague_schedule(source, move_t, selected)
            elif (
                self.args.profile == 'colleague-paper-closed'
                and self.args.is2_selection_mode == 'stable-window'
                and self.args.fixed_id_t_sec <= 0.0
                and self.args.fixed_id_rope_length_m <= 0.0
            ):
                self._abort_without_estimate(
                    move_t,
                    reason='No stable IS2 ID window passed the stability gates',
                )
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
            latest = self._latest_active_estimate()
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
        if not controller_timed_active:
            self._publish_stream(vx, vy)
        self._log_row(move_t, vx, vy)

        end_t = self._end_time()
        if move_t >= end_t and self.final_zero_wall is None:
            self.final_zero_wall = now
            self.phase = 'residual'
            if not controller_timed_active:
                self._publish_stream(0.0, 0.0)
            self.get_logger().info(
                f'Paper TDF command is zero; collecting residual swing for '
                f'{self.args.residual_window:.1f}s')
            return
        if self.final_zero_wall is not None:
            if not controller_timed_active:
                self._publish_stream(0.0, 0.0)
            if now - self.final_zero_wall >= self.args.residual_window:
                self._finish_run()
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
            latest = self._latest_active_estimate()
            if latest is None:
                self.get_logger().info('waiting for online ID estimate before first switch')
            else:
                self.get_logger().info(
                    f'latest ID before lock: t={latest.t:.3f}s T={latest.T:.4f}s '
                    f'zeta={latest.zeta:.4f} condB={latest.cond_b:.3e}')

    def _fixed_id_estimate(self, move_t: float) -> tuple[str, Estimate | None]:
        if self.args.fixed_id_t_sec > 0.0:
            T = self.args.fixed_id_t_sec
            omega_n = math.pi / T
            source = self.args.profile.replace('-', '_') + '_fixed_id_T'
        elif self.args.fixed_id_rope_length_m > 0.0:
            omega_n = math.sqrt(self.args.gravity_m_s2 / self.args.fixed_id_rope_length_m)
            T = math.pi / omega_n
            source = self.args.profile.replace('-', '_') + '_fixed_id_rope'
        else:
            return '', None

        zeta = self.args.fixed_id_zeta
        est = Estimate(
            t=move_t,
            x=0.0,
            i1=float('nan'),
            i2=float('nan'),
            i3=float('nan'),
            cond_b=0.0,
            omega_n=omega_n,
            zeta=zeta,
            T=T,
            A0=float('nan'),
            A1=float('nan'),
            A2=float('nan'),
        )
        return source, est

    def _latest_active_estimate(self) -> Estimate | None:
        if self.args.id_method == 'paper2-step':
            return self.latest_paper2_estimate
        if self.args.id_method == 'zero-zeta-ls':
            return self.latest_zero_zeta_estimate
        if self.args.id_method == 'freq-bank':
            return self.latest_freq_bank_estimate
        return self.identifier.latest_valid

    def _select_colleague_lock_estimate(self, move_t: float) -> tuple[str, Estimate | None]:
        source_base = self.args.profile.replace('-', '_')
        if self.args.id_method == 'paper2-step':
            source_base += '_paper2_step'
        elif self.args.id_method == 'zero-zeta-ls':
            source_base += '_zero_zeta_ls'
        elif self.args.id_method == 'freq-bank':
            source_base += '_freq_bank'
        if self.args.profile != 'colleague-paper-closed':
            if self.best_estimate is not None:
                return source_base + '_best_cond', self.best_estimate
            return source_base + '_latest', None

        candidates = [
            est
            for est in self.valid_estimates
            if est.t <= move_t and self._estimate_passes_colleague_id_gates(est)
        ]
        if self.args.is2_schedule_filter:
            before = len(candidates)
            candidates = [
                est
                for est in candidates
                if self._colleague_schedule_feasible_for_estimate(est)
            ]
            removed = before - len(candidates)
            if removed > 0:
                self.get_logger().info(
                    f'IS2 schedule feasibility filter removed {removed} of {before} candidates')
        if not candidates:
            return source_base + '_median_T', None
        if self.args.is2_selection_mode == 'stable-window':
            source, selected = self._select_stable_is2_window(source_base, move_t, candidates)
            if selected is not None:
                return source, selected
            return source, None
        if self.args.is2_selection_mode == 'latest':
            selected = max(candidates, key=lambda est: est.t)
            source = source_base + '_latest_T'
            self.get_logger().info(
                f'IS2 {source} selection: selected latest of {len(candidates)} '
                f'valid estimates; selected_t={selected.t:.3f}s '
                f'selected_T={selected.T:.4f}s condB={selected.cond_b:.3e}')
            return source, selected

        source = source_base + '_median_T'
        if (
            self.args.is2_selection_mode == 'recent-median'
            and self.args.is2_selection_window_s > 0.0
        ):
            window_start = max(0.0, move_t - self.args.is2_selection_window_s)
            recent = [est for est in candidates if est.t >= window_start]
            if recent:
                candidates = recent
                source = source_base + '_recent_median_T'
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
            f'IS2 {source} selection: median_T={median_T:.4f}s from '
            f'{len(candidates)} valid estimates; selected_t={selected.t:.3f}s '
            f'selected_T={selected.T:.4f}s')
        return source, selected

    def _select_stable_is2_window(
        self,
        source_base: str,
        move_t: float,
        candidates: list[Estimate],
    ) -> tuple[str, Estimate | None]:
        window_s = max(0.05, self.args.is2_stable_window_s)
        step_s = max(0.02, self.args.is2_stable_step_s)
        min_count = max(3, self.args.is2_stable_min_count)
        search_start = max(self.args.min_id_duration, self.args.is2_stable_after_s)
        usable = [est for est in candidates if search_start <= est.t <= move_t]
        if len(usable) < min_count:
            self.get_logger().warn(
                f'IS2 stable-window selection found only {len(usable)} usable estimates '
                f'after {search_start:.3f}s; falling back to median selection')
            return source_base + '_stable_window', None

        first_t = min(est.t for est in usable)
        last_t = max(est.t for est in usable)
        end = min(move_t, last_t)
        best: tuple[float, float, float, float, float, float, list[Estimate]] | None = None
        while end >= first_t + window_s:
            start = end - window_s
            win = [est for est in usable if start <= est.t <= end]
            if len(win) >= min_count:
                t_values = [est.t for est in win]
                T_values = sorted(est.T for est in win)
                mid = len(T_values) // 2
                median_T = (
                    T_values[mid]
                    if len(T_values) % 2 == 1
                    else 0.5 * (T_values[mid - 1] + T_values[mid])
                )
                q1 = T_values[len(T_values) // 4]
                q3 = T_values[(3 * (len(T_values) - 1)) // 4]
                spread = q3 - q1
                range_spread = T_values[-1] - T_values[0]
                if (
                    self.args.is2_stable_max_range_s > 0.0
                    and range_spread > self.args.is2_stable_max_range_s
                ):
                    end -= step_s
                    continue
                t_mean = sum(t_values) / len(t_values)
                T_mean = sum(est.T for est in win) / len(win)
                denom = sum((t - t_mean) ** 2 for t in t_values)
                slope = 0.0
                if denom > 1.0e-12:
                    slope = sum(
                        (est.t - t_mean) * (est.T - T_mean)
                        for est in win
                    ) / denom
                slope_span = abs(slope) * window_s
                median_cond = sorted(est.cond_b for est in win)[len(win) // 2]
                score = spread + self.args.is2_stable_slope_weight * slope_span
                # Prefer stable windows; use conditioning and later windows as tie-breakers.
                rank = (score, median_cond, -end)
                if best is None or rank < best[:3]:
                    best = (score, median_cond, -end, median_T, slope, spread, win)
            end -= step_s

        if best is None:
            self.get_logger().warn(
                'IS2 stable-window selection found no window that passed stability gates')
            return source_base + '_stable_window', None

        score, median_cond, neg_end, median_T, slope, iqr_spread, win = best
        selected = min(
            win,
            key=lambda est: (abs(est.T - median_T), -est.t, est.cond_b),
        )
        start = min(est.t for est in win)
        end = -neg_end
        range_spread = max(est.T for est in win) - min(est.T for est in win)
        self.get_logger().info(
            f'IS2 {source_base}_stable_window selection: '
            f'window=[{start:.3f},{end:.3f}]s n={len(win)} '
            f'median_T={median_T:.4f}s selected_t={selected.t:.3f}s '
            f'selected_T={selected.T:.4f}s iqr={iqr_spread:.4f}s '
            f'range={range_spread:.4f}s '
            f'slope={slope:.4f}s/s score={score:.4f} median_condB={median_cond:.3e}')
        return source_base + '_stable_window', selected

    def _abort_without_estimate(self, move_t: float, reason: str | None = None):
        if self.aborted:
            return
        self.aborted = True
        self.final_zero_wall = time.monotonic()
        if (
            getattr(self, 'controller_timed_profile_active', False)
            or getattr(self, 'timed_profile_start_wall', None) is not None
        ):
            self._publish_traj_abort()
        else:
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
            pulse_t = move_t - self.args.pulse_pre_delay_s
            return (
                self.direction * self.vmax_abs_mm_s
                if 0.0 <= pulse_t < self.tf
                else 0.0
            )

        sched = self.schedule
        if sched is None:
            return self.initial_velocity_mm_s

        if (
            self.args.profile == 'nonzero-ic'
            and self.nonzero_ic_profile_events is not None
        ):
            velocity_mm_s = 0.0
            for event_time_s, event_velocity_mm_s in self.nonzero_ic_profile_events:
                if move_t < event_time_s:
                    break
                velocity_mm_s = event_velocity_mm_s
            return self.direction * velocity_mm_s

        T = sched.T
        A0, A1 = sched.A0, sched.A1
        if self.args.profile in ('zv', 'nonzero-ic'):
            if move_t < T:
                gain = A0
            elif move_t < self.tf:
                gain = 1.0
            elif move_t < self.tf + T:
                gain = 1.0 - A0
            else:
                gain = 0.0
            if self.args.profile == 'nonzero-ic':
                return self.direction * self.vmax_abs_mm_s * gain
            return self.direction * self.vmax_abs_mm_s * max(0.0, gain)

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
        if self.args.is2_id_t_min > 0.0 and est.T < self.args.is2_id_t_min:
            return False
        if self.args.is2_id_t_max > 0.0 and est.T > self.args.is2_id_t_max:
            return False
        return self.args.id_zeta_min <= est.zeta <= self.args.id_zeta_max

    def _end_time(self) -> float:
        if self.args.profile == 'pulse':
            return self.args.pulse_pre_delay_s + self.tf
        if self.args.profile in ('zv', 'nonzero-ic'):
            T = self.args.fallback_t if self.schedule is None else self.schedule.T
            return self.tf + T
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

        if self.excite_only_complete:
            status = 'Bounded excitation-only trial complete'
        elif self.args.profile == 'pulse':
            status = 'Pulse baseline move complete'
        else:
            status = 'Paper TDF move aborted' if self.aborted else 'Paper TDF move complete'
        excite_text = ''
        if self.exciter is not None and self._latest_excite_cmd is not None:
            cmd = self._latest_excite_cmd
            excite_text = (
                f' | excite: A_est={cmd.amplitude_est_deg:.1f}deg '
                f'(target {self.args.excite_target_angle_deg:.1f}deg) '
                f'drive_sign={cmd.drive_sign:+.0f}'
            )
        if self.excite_only_complete:
            anchor_error = (
                float('nan')
                if self.latest_cart_q_mm is None or self.excite_start_cart_q_mm is None
                else self.latest_cart_q_mm - self.excite_start_cart_q_mm
            )
            travel_text = f'excitation anchor return error={anchor_error:+.2f} mm'
        else:
            travel_text = (
                f'travel: target={self.args.target_distance_mm:.1f} mm '
                f'actual={traveled:.1f} mm error={target_error:+.1f} mm')
        self.get_logger().info(
            f'{status} | {travel_text} | '
            f'{sample_text} | {estimate_text}{id_text}{excite_text}')
        if self.args.profile == 'nonzero-ic':
            if self.timed_profile_payload_clock_offset_s is not None:
                arm_queue_text = (
                    'unavailable'
                    if self.timed_profile_arm_payload_queue_delay_ms is None
                    else (
                        f'{self.timed_profile_arm_payload_queue_delay_ms:.2f}ms'
                    )
                )
                self.get_logger().info(
                    'Payload timing diagnostics: calibrated clock offset='
                    f'{self.timed_profile_payload_clock_offset_s:.6f}s, '
                    f'queue delay at arming={arm_queue_text}, gate='
                    f'{self.args.nonzero_ic_max_payload_queue_delay_ms:.2f}ms')
            command_observation = self.nonzero_ic_measured_command_start
            peak_observation = self.nonzero_ic_measured_peak
            if command_observation is not None or peak_observation is not None:
                command_text = (
                    'unavailable'
                    if command_observation is None
                    else (
                        f'v={command_observation.world_payload_velocity_mm_s:+.2f}'
                        f'mm/s sample_dt={command_observation.offset_ms:+.2f}ms'
                    )
                )
                peak_text = (
                    'unavailable'
                    if peak_observation is None
                    else (
                        f'v={peak_observation.world_payload_velocity_mm_s:+.2f}'
                        f'mm/s sample_dt={peak_observation.offset_ms:+.2f}ms'
                    )
                )
                self.get_logger().info(
                    'Peak-state diagnostics (measured velocity is filtered '
                    'encoder-relative velocity plus measured cart velocity): '
                    f'predicted_peak_v='
                    f'{self.nonzero_ic_predicted_peak_payload_velocity_mm_s:+.3f}'
                    f'mm/s | measured_near_command_start={command_text} | '
                    f'measured_near_intended_physical_peak={peak_text}'
                )

    def _finish_run(self):
        if self.done:
            return
        self.done = True
        self._print_final_report()
        if self.csv_file is not None:
            self.csv_file.flush()
            self._generate_run_outputs()

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

    def _publish_traj_abort(self):
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = TrajCmd.ABORT
        self.traj_pub.publish(msg)

    def _log_row(self, move_t: float, vx: float, vy: float):
        if self.csv_writer is None:
            return
        log_wall = time.monotonic()
        latest = self._latest_active_estimate()
        logged_omega_n = (
            latest.omega_n
            if latest is not None
            else self.nonzero_ic_used_omega_rad_s
        )
        nonzero_fit = self.nonzero_ic_frequency_estimate
        candidate = self.identifier.latest_candidate
        sched = self.schedule
        robust = self.nonzero_ic_robust_solution
        source = 'pulse' if self.args.profile == 'pulse' else ('' if sched is None else sched.source)
        cart_sample = self.latest_cart_state_sample or self.latest_gantry_state_sample
        state_age_ms = (
            ''
            if self.latest_gantry_state_sample is None
            else 1000.0 * max(0.0, log_wall - self.latest_gantry_state_sample.rx_wall)
        )
        payload_age_ms = (
            ''
            if self.latest_payload_wall is None
            else 1000.0 * max(0.0, log_wall - self.latest_payload_wall)
        )
        latency_values = (
            [''] * 26
            if self.latest_traj_latency is None
            else self.latest_traj_latency
        )
        latency_age_ms = (
            ''
            if self.latest_traj_latency_wall is None
            else 1000.0 * max(0.0, log_wall - self.latest_traj_latency_wall)
        )
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds * 1.0e-9,
            move_t,
            vx,
            vy,
            '' if self.latest_payload_abs_mm is None else self.latest_payload_abs_mm,
            '' if self.latest_swing_mm is None else self.latest_swing_mm,
            '' if self.latest_enc_pitch_deg is None else self.latest_enc_pitch_deg,
            '' if self.latest_enc_roll_deg is None else self.latest_enc_roll_deg,
            (
                ''
                if (self.latest_enc_pitch_deg is None and self.args.axis == 'x') or
                (self.latest_enc_roll_deg is None and self.args.axis == 'y')
                else (
                    self.latest_enc_pitch_deg
                    if self.args.axis == 'x'
                    else self.latest_enc_roll_deg
                )
            ),
            '' if self.latest_id_filtered_swing_mm is None else self.latest_id_filtered_swing_mm,
            '' if self.latest_cart_q_mm is None else self.latest_cart_q_mm,
            (
                ''
                if cart_sample is None
                else 1000.0 * float(cart_sample.vx)
            ),
            (
                ''
                if cart_sample is None
                else 1000.0 * float(cart_sample.vy)
            ),
            '' if cart_sample is None else cart_sample.stamp,
            state_age_ms,
            '' if self.latest_payload_time is None else self.latest_payload_time,
            payload_age_ms,
            (
                ''
                if self.latest_payload_observed_clock_offset_s is None
                else self.latest_payload_observed_clock_offset_s
            ),
            (
                ''
                if self.payload_clock_offset_s is None
                else self.payload_clock_offset_s
            ),
            (
                ''
                if self.latest_payload_queue_delay_ms is None
                else self.latest_payload_queue_delay_ms
            ),
            (
                ''
                if self.latest_payload_encoder_sample_age_ms is None
                else self.latest_payload_encoder_sample_age_ms
            ),
            (
                ''
                if self.latest_payload_rel_vx_mm_s is None
                else self.latest_payload_rel_vx_mm_s
            ),
            (
                ''
                if self.latest_payload_rel_raw_vx_mm_s is None
                else self.latest_payload_rel_raw_vx_mm_s
            ),
            '' if self.latest_stream_dt_ms is None else self.latest_stream_dt_ms,
            self._traveled_target_distance_mm(),
            '' if latest is None else latest.cond_b,
            '' if logged_omega_n is None else logged_omega_n,
            (
                ''
                if self.nonzero_ic_fitted_omega_rad_s is None
                else self.nonzero_ic_fitted_omega_rad_s
            ),
            (
                ''
                if self.nonzero_ic_small_angle_omega_rad_s is None
                else self.nonzero_ic_small_angle_omega_rad_s
            ),
            (
                ''
                if self.nonzero_ic_fitted_amplitude_angle_deg is None
                else self.nonzero_ic_fitted_amplitude_angle_deg
            ),
            (
                ''
                if self.nonzero_ic_frequency_correction_factor is None
                else self.nonzero_ic_frequency_correction_factor
            ),
            (
                ''
                if self.nonzero_ic_used_omega_rad_s is None
                else self.nonzero_ic_used_omega_rad_s
            ),
            (
                ''
                if self.args.profile != 'nonzero-ic'
                else self.args.nonzero_ic_shaper_frequency_scale
            ),
            (
                ''
                if self.args.profile != 'nonzero-ic'
                else (
                    'fixed'
                    if self.args.nonzero_ic_shaper_omega_rad_s > 0.0
                    else (
                        'amplitude-corrected-scaled-fit'
                        if self.args.nonzero_ic_finite_amplitude_correction
                        else 'scaled-fit'
                    )
                )
            ),
            (
                ''
                if self.args.profile != 'nonzero-ic'
                else self.args.nonzero_ic_shaper_omega_rad_s
            ),
            int(robust is not None),
            '' if robust is None else robust.frequency_band_fraction,
            (
                '' if robust is None else ';'.join(
                    f'{value:.12g}' for value in robust.impulse_times_s)
            ),
            (
                '' if robust is None else ';'.join(
                    f'{value:.12g}' for value in robust.start_amplitudes)
            ),
            (
                '' if robust is None else ';'.join(
                    f'{value:.12g}' for value in robust.stop_amplitudes)
            ),
            '' if robust is None else robust.worst_case_residual_fraction,
            (
                '' if robust is None
                else robust.baseline_worst_case_residual_fraction
            ),
            '' if robust is None else robust.optimizer_iterations,
            (
                ''
                if self.args.profile != 'nonzero-ic'
                else self.args.timed_profile_actuation_lead_ms
            ),
            (
                ''
                if self.timed_profile_peak_wall is None
                else self.timed_profile_peak_wall
            ),
            (
                ''
                if self.timed_profile_start_wall is None
                else self.timed_profile_start_wall
            ),
            (
                ''
                if self.timed_profile_peak_payload_time is None
                else self.timed_profile_peak_payload_time
            ),
            (
                ''
                if self.timed_profile_start_payload_time is None
                else self.timed_profile_start_payload_time
            ),
            (
                ''
                if self.timed_profile_payload_clock_offset_s is None
                else self.timed_profile_payload_clock_offset_s
            ),
            (
                ''
                if self.timed_profile_arm_payload_queue_delay_ms is None
                else self.timed_profile_arm_payload_queue_delay_ms
            ),
            (
                ''
                if self.nonzero_ic_predicted_peak_swing_mm is None
                else self.nonzero_ic_predicted_peak_swing_mm
            ),
            (
                ''
                if self.nonzero_ic_predicted_peak_payload_velocity_mm_s is None
                else self.nonzero_ic_predicted_peak_payload_velocity_mm_s
            ),
            (
                ''
                if self.nonzero_ic_predicted_command_start_swing_mm is None
                else self.nonzero_ic_predicted_command_start_swing_mm
            ),
            (
                ''
                if self.nonzero_ic_predicted_command_start_payload_velocity_mm_s is None
                else self.nonzero_ic_predicted_command_start_payload_velocity_mm_s
            ),
            (
                ''
                if self.nonzero_ic_measured_command_start is None
                else self.nonzero_ic_measured_command_start.swing_mm
            ),
            (
                ''
                if self.nonzero_ic_measured_command_start is None
                else self.nonzero_ic_measured_command_start.world_payload_velocity_mm_s
            ),
            (
                ''
                if self.nonzero_ic_measured_command_start is None
                else self.nonzero_ic_measured_command_start.offset_ms
            ),
            (
                ''
                if self.nonzero_ic_measured_peak is None
                else self.nonzero_ic_measured_peak.swing_mm
            ),
            (
                ''
                if self.nonzero_ic_measured_peak is None
                else self.nonzero_ic_measured_peak.world_payload_velocity_mm_s
            ),
            (
                ''
                if self.nonzero_ic_measured_peak is None
                else self.nonzero_ic_measured_peak.offset_ms
            ),
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
            (
                ''
                if self.nonzero_ic_initial_swing_mm is None
                else self.nonzero_ic_initial_swing_mm
            ),
            (
                ''
                if self.nonzero_ic_initial_payload_velocity_mm_s is None
                else self.nonzero_ic_initial_payload_velocity_mm_s
            ),
            '' if nonzero_fit is None else nonzero_fit.amplitude_mm,
            '' if nonzero_fit is None else nonzero_fit.rmse_mm,
            '' if nonzero_fit is None else nonzero_fit.normalized_rmse,
            '' if nonzero_fit is None else nonzero_fit.sample_count,
            '' if nonzero_fit is None else nonzero_fit.window_duration_s,
            '' if candidate is None else candidate.t,
            '' if candidate is None else candidate.cond_b,
            '' if candidate is None else candidate.omega_n,
            '' if candidate is None else candidate.zeta,
            '' if candidate is None else candidate.T,
            '' if candidate is None else int(candidate.valid),
            '' if candidate is None else candidate.reject_reason,
            self.args.eq21_input_model,
            '' if self.latest_zero_zeta_estimate is None else self.latest_zero_zeta_estimate.T,
            '' if self.latest_zero_zeta_score is None else self.latest_zero_zeta_score,
            '' if self.latest_zero_zeta_rmse is None else self.latest_zero_zeta_rmse,
            '' if self.latest_zero_zeta_amp is None else self.latest_zero_zeta_amp,
            self.args.final_id_source,
            self.args.tau,
            abs(self.initial_velocity_mm_s),
            '' if self.latest_two_mode_t2 is None else self.latest_two_mode_t2,
            '' if self.latest_two_mode_amp1 is None else self.latest_two_mode_amp1,
            '' if self.latest_two_mode_amp2 is None else self.latest_two_mode_amp2,
            '' if self.latest_two_mode_amp_ratio is None else self.latest_two_mode_amp_ratio,
            '' if self.latest_two_mode_nrmse is None else self.latest_two_mode_nrmse,
            '' if self.latest_enc_pitch_count is None else self.latest_enc_pitch_count,
            '' if self.latest_enc_roll_count is None else self.latest_enc_roll_count,
            '' if self.latest_payload_rel_x_mm is None else self.latest_payload_rel_x_mm,
            '' if self.latest_payload_rel_y_mm is None else self.latest_payload_rel_y_mm,
            '' if self.latest_payload_rel_z_mm is None else self.latest_payload_rel_z_mm,
            '' if self.latest_enc_arduino_ms is None else self.latest_enc_arduino_ms,
            '' if self.latest_enc_pitch_raw is None else self.latest_enc_pitch_raw,
            '' if self.latest_enc_roll_raw is None else self.latest_enc_roll_raw,
            '' if self.latest_enc_imu1_ax is None else self.latest_enc_imu1_ax,
            '' if self.latest_enc_imu1_ay is None else self.latest_enc_imu1_ay,
            '' if self.latest_enc_imu1_az is None else self.latest_enc_imu1_az,
            '' if self.latest_enc_imu1_gx is None else self.latest_enc_imu1_gx,
            '' if self.latest_enc_imu1_gy is None else self.latest_enc_imu1_gy,
            '' if self.latest_enc_imu1_gz is None else self.latest_enc_imu1_gz,
            '' if self.latest_enc_imu2_ax is None else self.latest_enc_imu2_ax,
            '' if self.latest_enc_imu2_ay is None else self.latest_enc_imu2_ay,
            '' if self.latest_enc_imu2_az is None else self.latest_enc_imu2_az,
            '' if self.latest_enc_imu2_gx is None else self.latest_enc_imu2_gx,
            '' if self.latest_enc_imu2_gy is None else self.latest_enc_imu2_gy,
            '' if self.latest_enc_imu2_gz is None else self.latest_enc_imu2_gz,
            '' if self.latest_enc_packet_age_ms is None else self.latest_enc_packet_age_ms,
            '' if self.latest_enc_packet_seen is None else self.latest_enc_packet_seen,
            '' if self.latest_enc_serial_lines is None else self.latest_enc_serial_lines,
            '' if self.latest_enc_parse_errors is None else self.latest_enc_parse_errors,
            '' if self.latest_enc_stale is None else self.latest_enc_stale,
            '' if self.latest_cam_gantry_x_mm is None else self.latest_cam_gantry_x_mm,
            (
                ''
                if self.latest_cam_gantry_x_mm is None or cart_sample is None
                else self.latest_cam_gantry_x_mm - 1000.0 * float(cart_sample.x)
            ),
            '' if self.latest_cam_gantry_y_mm is None else self.latest_cam_gantry_y_mm,
            (
                ''
                if self.latest_cam_gantry_y_mm is None or cart_sample is None
                else self.latest_cam_gantry_y_mm - 1000.0 * float(cart_sample.y)
            ),
            *latency_values,
            latency_age_ms,
            self.phase,
            '' if self.run_start_wall is None else log_wall - self.run_start_wall,
            *self._excite_log_values(),
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
            'zv',
            'robust',
            'nonzero-ic',
        ])
    parser.add_argument('--payload-topic', default='/payload/pose_e_rel')
    parser.add_argument('--target-distance-mm', type=float, default=600.0)
    parser.add_argument('--vmax-mm-s', type=float, default=200.0)
    parser.add_argument(
        '--pulse-pre-delay-s',
        type=float,
        default=0.0,
        help=(
            'For --profile pulse: hold STREAM at zero for this duration after '
            'MOTION_START before applying the velocity step.'
        ),
    )
    parser.add_argument('--a0', type=float, default=0.5)
    parser.add_argument(
        '--tau',
        type=float,
        default=None,
        help=(
            'For colleague profiles: closed-form timing offset. For '
            '--profile nonzero-ic: telemetry-ready countdown and adaptive '
            'free-swing observation window (default: 5.0s for nonzero-ic, '
            '0.75s otherwise).'
        ))
    parser.add_argument(
        '--id-duration-s',
        type=float,
        default=0.0,
        help='Alias for --tau for fixed-duration ID tests. <=0 disables alias.')
    parser.add_argument(
        '--id-speed-mm-s',
        type=float,
        default=0.0,
        help='Alias for --a0 using id_speed/vmax for fixed-speed ID tests. <=0 disables alias.')
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
        '--id-method',
        choices=('integral', 'paper2-step', 'zero-zeta-ls', 'freq-bank'),
        default='integral',
        help='Online ID method used for adaptive locking.')
    parser.add_argument(
        '--eq21-input-model',
        choices=('measured_cart', 'ideal_k'),
        default='measured_cart',
        help='Eq21 input model: measured_cart uses measured q(t); ideal_k assumes q(t)=K*t.')
    parser.add_argument(
        '--final-id-source',
        choices=('active_id', 'zero_zeta'),
        default='active_id',
        help='Final ID selector. zero_zeta maps to --id-method zero-zeta-ls for conservative fixed-tau tests.')
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
    parser.add_argument('--workspace-min-mm', type=float, default=0.0)
    parser.add_argument('--workspace-max-mm', type=float, default=1150.0)
    parser.add_argument('--workspace-margin-mm', type=float, default=5.0)
    parser.add_argument('--stream-rate-hz', type=float, default=100.0)
    parser.add_argument('--print-period', type=float, default=0.25)
    parser.add_argument('--payload-fresh-timeout', type=float, default=0.25)
    parser.add_argument(
        '--gantry-fresh-timeout', type=float, default=0.25,
        help='Maximum /gantry/state receive age allowed before start and during nonzero-IC motion.')
    parser.add_argument(
        '--require-encoder-health',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Require fresh /payload/encoder/diagnostics with stale=0 for '
            'nonzero-IC experiments (default: enabled). Disable only when the '
            'payload topic intentionally comes from a non-encoder source.'
        ),
    )
    parser.add_argument('--cond-threshold', type=float, default=1.0e8)
    parser.add_argument('--accept-cond', type=float, default=1000.0)
    parser.add_argument('--accept-valid-count', type=int, default=1)
    parser.add_argument('--stability-count', type=int, default=1)
    parser.add_argument('--stability-tol', type=float, default=0.08)
    parser.add_argument('--omega-min', type=float, default=0.5)
    parser.add_argument('--omega-max', type=float, default=20.0)
    parser.add_argument(
        '--id-lowpass-hz',
        type=float,
        default=0.0,
        help='First-order low-pass cutoff for the ID input signal. <=0 disables filtering.')
    parser.add_argument(
        '--integral-id-window-s',
        type=float,
        default=0.0,
        help='Sliding local window for integral ID. <=0 uses full-history Eq. 21 ID.')
    parser.add_argument(
        '--id-assume-zero-zeta',
        action='store_true',
        help='For integral ID, estimate omega_n assuming zeta=0 instead of fitting damping.')
    parser.add_argument('--mode-fit-window-s', type=float, default=1.00)
    parser.add_argument('--mode-fit-history-s', type=float, default=2.00)
    parser.add_argument('--mode-fit-after-s', type=float, default=0.80)
    parser.add_argument('--mode-fit-min-samples', type=int, default=40)
    parser.add_argument('--mode-fit-t-min', type=float, default=0.60)
    parser.add_argument('--mode-fit-t-max', type=float, default=1.25)
    parser.add_argument('--mode-fit-grid-count', type=int, default=160)
    parser.add_argument('--mode-fit-min-p2p-mm', type=float, default=8.0)
    parser.add_argument('--mode-fit-min-amp-mm', type=float, default=3.0)
    parser.add_argument('--mode-fit-max-norm-rmse', type=float, default=0.85)
    parser.add_argument('--mode-fit-cond-weight', type=float, default=0.01)
    parser.add_argument(
        '--mode-fit-edge-margin-s',
        type=float,
        default=0.02,
        help='Reject mode-fit estimates this close to the T search bounds.')
    parser.add_argument('--zero-zeta-window-s', type=float, default=0.0)
    parser.add_argument('--zero-zeta-history-s', type=float, default=4.0)
    parser.add_argument('--zero-zeta-after-s', type=float, default=0.8)
    parser.add_argument('--zero-zeta-update-period-s', type=float, default=0.10)
    parser.add_argument('--zero-zeta-min-samples', type=int, default=80)
    parser.add_argument('--zero-zeta-t-min', type=float, default=0.45)
    parser.add_argument('--zero-zeta-t-max', type=float, default=1.45)
    parser.add_argument('--zero-zeta-grid-count', type=int, default=240)
    parser.add_argument('--zero-zeta-min-p2p-mm', type=float, default=4.0)
    parser.add_argument('--zero-zeta-min-amp-mm', type=float, default=1.0)
    parser.add_argument('--zero-zeta-max-norm-rmse', type=float, default=1.5)
    parser.add_argument(
        '--freq-bank-update-period-s',
        type=float,
        default=0.10,
        help='Minimum time between frequency-bank ID updates.')
    parser.add_argument(
        '--freq-bank-min-ratio',
        type=float,
        default=1.03,
        help='Minimum best/second-best coherent-power ratio for frequency-bank ID.')
    parser.add_argument('--two-mode-window-s', type=float, default=1.20)
    parser.add_argument('--two-mode-history-s', type=float, default=2.00)
    parser.add_argument('--two-mode-after-s', type=float, default=0.90)
    parser.add_argument('--two-mode-update-period-s', type=float, default=0.05)
    parser.add_argument('--two-mode-min-samples', type=int, default=55)
    parser.add_argument('--two-mode-t1-min', type=float, default=0.50)
    parser.add_argument('--two-mode-t1-max', type=float, default=1.35)
    parser.add_argument('--two-mode-t1-grid-count', type=int, default=80)
    parser.add_argument('--two-mode-t2-min', type=float, default=0.12)
    parser.add_argument('--two-mode-t2-max', type=float, default=0.80)
    parser.add_argument('--two-mode-t2-grid-count', type=int, default=55)
    parser.add_argument('--two-mode-max-t2-t1-ratio', type=float, default=0.75)
    parser.add_argument('--two-mode-min-amp1-mm', type=float, default=4.0)
    parser.add_argument('--two-mode-max-norm-rmse', type=float, default=0.75)
    parser.add_argument('--two-mode-cond-weight', type=float, default=0.01)
    parser.add_argument(
        '--two-mode-amp2-weight',
        type=float,
        default=0.0,
        help='Optional score penalty on amp2/amp1. Default 0 keeps second mode diagnostic.')
    parser.add_argument('--paper2-window-s', type=float, default=1.60)
    parser.add_argument('--paper2-history-s', type=float, default=2.50)
    parser.add_argument('--paper2-after-s', type=float, default=0.80)
    parser.add_argument('--paper2-min-samples', type=int, default=40)
    parser.add_argument('--paper2-min-p2p-mm', type=float, default=8.0)
    parser.add_argument('--paper2-min-peak-mm', type=float, default=3.0)
    parser.add_argument('--paper2-min-peak-dt', type=float, default=0.20)
    parser.add_argument('--paper2-min-extrema', type=int, default=3)
    parser.add_argument('--paper2-smooth-radius', type=int, default=2)
    parser.add_argument(
        '--is2-selection-window-s',
        type=float,
        default=0.5,
        help='For colleague-paper-closed, choose median T from this recent pre-tau window. <=0 uses all valid estimates.')
    parser.add_argument(
        '--is2-selection-mode',
        choices=('median', 'recent-median', 'stable-window', 'stable', 'latest'),
        default='recent-median',
        help='Estimate selector for colleague-paper-closed.')
    parser.add_argument(
        '--is2-stable-window-s',
        type=float,
        default=0.50,
        help='Sliding window duration for stable-window IS2 selection.')
    parser.add_argument(
        '--is2-stable-step-s',
        type=float,
        default=0.05,
        help='Sliding window step for stable-window IS2 selection.')
    parser.add_argument(
        '--is2-stable-after-s',
        type=float,
        default=1.00,
        help='Ignore IS2 stable-window candidates before this ID time.')
    parser.add_argument(
        '--is2-stable-min-count',
        type=int,
        default=12,
        help='Minimum valid estimates required inside a stable-window candidate.')
    parser.add_argument(
        '--is2-stable-slope-weight',
        type=float,
        default=1.0,
        help='Weight on abs(T slope)*window in stable-window score.')
    parser.add_argument(
        '--is2-stable-max-range-s',
        type=float,
        default=0.45,
        help='Reject stable-window candidates whose max(T)-min(T) exceeds this. <=0 disables.')
    parser.add_argument(
        '--is2-id-t-min',
        type=float,
        default=0.0,
        help='Reject IS2 ID candidates with estimated ID_T below this. <=0 disables.')
    parser.add_argument(
        '--is2-id-t-max',
        type=float,
        default=0.0,
        help='Reject IS2 ID candidates with estimated ID_T above this. <=0 disables.')
    parser.add_argument(
        '--is2-schedule-filter',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Filter IS2 candidates by whether they produce a feasible Paper-2 schedule.')
    parser.add_argument(
        '--is2-schedule-margin-s',
        type=float,
        default=0.50,
        help='Minimum required gap tf - (tau + 2*schedule_T) for IS2 candidate schedules. <=0 disables margin.')
    parser.add_argument(
        '--fixed-id-t-sec',
        type=float,
        default=0.0,
        help='Bypass online ID and inject this shaper ID period T [s] at tau. <=0 disables.')
    parser.add_argument(
        '--fixed-id-rope-length-m',
        type=float,
        default=0.0,
        help='Bypass online ID using omega=sqrt(g/L) and T=pi/omega for this rope length. <=0 disables.')
    parser.add_argument(
        '--fixed-id-zeta',
        type=float,
        default=0.0,
        help='Damping ratio attached to the injected fixed ID estimate.')
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
        help='For --profile robust/zv/nonzero-ic: nominal rope length used for model-based shaping.')
    parser.add_argument(
        '--robust-zeta',
        type=float,
        default=0.0,
        help='For --profile robust/zv: damping ratio used for model-based shaper weights.')
    parser.add_argument(
        '--robust-t-scale',
        type=float,
        default=1.0,
        help='For --profile robust/zv: multiplier on T=pi/sqrt(g/L).')
    parser.add_argument(
        '--gravity-m-s2',
        type=float,
        default=9.80665,
        help='Gravity used by model-based profiles.')
    parser.add_argument(
        '--nonzero-ic-omega-rad-s',
        type=float,
        default=0.0,
        help=(
            'Fixed undamped natural frequency used with '
            '--no-nonzero-ic-adaptive-frequency. <=0 uses '
            'sqrt(gravity/robust-rope-length-m).'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-adaptive-frequency',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Fit omega and the instantaneous initial state from the latest '
            'tau seconds of free swing (default: enabled).'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-shaper-frequency-scale',
        type=float,
        default=1.0,
        help=(
            'Experimental multiplier applied only to the nonzero-IC shaper '
            'frequency. The fitted frequency and future-peak prediction '
            'remain unchanged (default: 1.0).'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-finite-amplitude-correction',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'For adaptive nonzero-IC only: convert the fitted finite-amplitude '
            'free-swing frequency to the simple-pendulum small-angle frequency '
            'using the fitted displacement amplitude and '
            '--robust-rope-length-m. The uncorrected fit remains in use for '
            'state and future-peak prediction (default: disabled).'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-robust',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Replace the two-impulse NZIC staircase with a positive-weight '
            'specified-insensitivity schedule optimized over '
            '--nonzero-ic-robust-band-fraction while preserving nominal '
            'cancellation and exact travel (default: disabled).'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-robust-band-fraction',
        type=float,
        default=0.05,
        help=(
            'Fractional +/- frequency band for robust nonzero-IC schedule '
            'optimization (default: 0.05, or +/-5%%).'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-shaper-omega-rad-s',
        type=float,
        default=0.0,
        help=(
            'Absolute experimental frequency used only by the nonzero-IC '
            'closed-form shaper. A positive value overrides the fitted '
            'frequency for schedule construction while adaptive state/peak '
            'estimation remains active; 0 disables the override (default: 0).'
        ),
    )
    parser.add_argument('--nonzero-ic-omega-min-rad-s', type=float, default=2.6)
    parser.add_argument('--nonzero-ic-omega-max-rad-s', type=float, default=4.0)
    parser.add_argument('--nonzero-ic-frequency-grid-count', type=int, default=401)
    parser.add_argument('--nonzero-ic-min-fit-samples', type=int, default=100)
    parser.add_argument('--nonzero-ic-min-fit-amplitude-mm', type=float, default=10.0)
    parser.add_argument(
        '--nonzero-ic-max-fit-nrmse',
        type=float,
        default=0.05,
        help=(
            'Maximum RMSE/fitted-amplitude accepted for adaptive free-swing '
            'ID (default: 0.05).'
        ),
    )
    parser.add_argument('--nonzero-ic-fit-update-period-s', type=float, default=0.25)
    parser.add_argument(
        '--controller-timed-profile',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'For adaptive nonzero-IC, atomically load the solved staircase '
            'into the gantry controller and start it at a predicted future '
            'positive peak (default: enabled).'
        ),
    )
    parser.add_argument(
        '--timed-profile-lead-s',
        type=float,
        default=0.25,
        help='Minimum lead used when selecting the controller-owned future peak start.')
    parser.add_argument(
        '--timed-profile-actuation-lead-ms',
        type=float,
        default=0.0,
        help=(
            'Equivalent actuator-response delay compensation: advance every '
            'controller-profile command edge by this many milliseconds so '
            'the modeled physical velocity response aligns with the fitted '
            'payload peak (default: 0). This pure-delay model does not alter '
            'the fitted state, gains, or shaper frequency.'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-start-at-peak',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Hold zero until the fitted payload is at its positive swing peak '
            'in the direction of travel (default: enabled).'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-peak-velocity-tolerance-mm-s',
        type=float,
        default=10.0,
        help=(
            'Maximum fitted absolute payload velocity accepted as the swing '
            'peak (default: 10 mm/s). A tighter value may wait additional '
            'swing cycles at a 100 Hz command rate. With the controller-timed '
            'profile this gate applies to the predicted future-peak state; '
            'nearest measured command/peak states are diagnostic only.'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-max-payload-queue-delay-ms',
        type=float,
        default=5.0,
        help=(
            'Maximum publisher-to-callback queue delay accepted while '
            'arming a controller-timed nonzero-IC profile (default: 5 ms). '
            'Delay is measured relative to the minimum observed payload '
            'clock offset; stale queued samples are held and retried.'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-start-delay-s',
        type=float,
        default=None,
        help=(
            'Deprecated alias for --tau in --profile nonzero-ic.'
        ),
    )
    parser.add_argument(
        '--nonzero-ic-max-command-speed-mm-s',
        type=float,
        default=400.0,
        help=(
            'For --profile nonzero-ic: maximum absolute velocity allowed for '
            'signed A0/A1 commands (default: 400 mm/s).'
        ),
    )
    parser.add_argument(
        '--initial-swing-mm',
        type=float,
        default=None,
        help=(
            'For --profile nonzero-ic: world-axis payload-minus-cart position '
            'at MOTION_START. Omit to use /payload/pose_e_rel.'
        ),
    )
    parser.add_argument(
        '--initial-payload-velocity-mm-s',
        type=float,
        default=None,
        help=(
            'For --profile nonzero-ic: world-axis absolute payload velocity '
            'at MOTION_START. Omit to use relative payload velocity plus cart '
            'velocity.'
        ),
    )
    parser.add_argument(
        '--excite',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'For --profile nonzero-ic: run a closed-loop swing-up stage that '
            'pumps the payload to --excite-target-angle-deg of axis swing '
            'before the tau free-swing ID window (default: disabled).'
        ),
    )
    parser.add_argument(
        '--excite-only',
        action='store_true',
        help=(
            'Run the bounded swing-up, return to its anchor, collect the '
            'residual window, and exit without the nonzero-IC travel maneuver.'
        ),
    )
    parser.add_argument('--excite-target-angle-deg', type=float, default=21.0)
    parser.add_argument('--excite-angle-tolerance-deg', type=float, default=1.5)
    parser.add_argument('--excite-angle-band-deg', type=float, default=5.0)
    parser.add_argument(
        '--excite-speed-mm-s',
        type=float,
        default=150.0,
        help='Peak trolley speed used by the swing-up drive. Keep well under '
             'abs(--vmax-mm-s).')
    parser.add_argument(
        '--excite-drive-sign',
        type=int,
        choices=(1, -1),
        default=1,
        help='Deprecated compatibility option; bounded excitation is always '
             'in the positive axis direction.')
    parser.add_argument('--excite-no-auto-sign', action='store_true')
    parser.add_argument('--excite-no-auto-bias', action='store_true',
                        help='Do not subtract the axis angle sampled at '
                             'MOTION_START as the swing-up zero reference.')
    parser.add_argument(
        '--excite-travel-budget-mm', type=float, default=100.0,
        help='Maximum positive-axis excursion from the excitation anchor.')
    parser.add_argument('--excite-initial-excursion-mm', type=float, default=15.0)
    parser.add_argument('--excite-excursion-step-mm', type=float, default=10.0)
    parser.add_argument('--excite-position-kp-s', type=float, default=2.0)
    parser.add_argument('--excite-return-speed-mm-s', type=float, default=30.0)
    parser.add_argument('--excite-return-tolerance-mm', type=float, default=1.0)
    parser.add_argument('--excite-slew-mm-s2', type=float, default=600.0)
    parser.add_argument('--excite-timeout-s', type=float, default=30.0)
    parser.add_argument('--excite-settle-cycles', type=float, default=1.0)
    parser.add_argument('--excite-standclear-s', type=float, default=2.0)
    parser.add_argument('--excite-abort-angle-deg', type=float, default=30.0)
    parser.add_argument(
        '--excite-omega-rad-s',
        type=float,
        default=0.0,
        help='Pendulum frequency for the swing-up envelope estimate. '
             '<=0 uses sqrt(gravity/--robust-rope-length-m).')
    parser.set_defaults(allow_fallback=True)
    parser.add_argument(
        '--no-fallback',
        dest='allow_fallback',
        action='store_false',
        help='Require an adaptive estimate; abort the STREAM if none is available by estimate-deadline.')
    parser.add_argument('--residual-window', type=float, default=10.0)
    parser.add_argument('--log-csv', default='')
    parser.add_argument(
        '--run-output-dir',
        default='~/crane_ws/log/adaptive_runs',
        help='Directory used for the automatically named CSV when --log-csv is omitted.')
    parser.add_argument(
        '--auto-plot',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Save a per-run PNG and JSON summary beside the CSV (default: enabled).')
    parser.add_argument('--no-arm', action='store_true')
    parser.add_argument('--no-auto-enable', action='store_true')
    parser.add_argument('--estimate-topic', default='/adaptive_paper_tdf/estimate')
    args = parser.parse_args()
    if args.is2_selection_mode == 'stable':
        args.is2_selection_mode = 'stable-window'
    if args.nonzero_ic_start_delay_s is not None:
        if args.tau is not None and not math.isclose(
            args.tau,
            args.nonzero_ic_start_delay_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            parser.error(
                '--nonzero-ic-start-delay-s and --tau specify different values'
            )
        args.tau = args.nonzero_ic_start_delay_s
    if args.tau is None:
        args.tau = 5.0 if args.profile == 'nonzero-ic' else 0.75
    if args.id_duration_s > 0.0:
        args.tau = args.id_duration_s
    if args.id_speed_mm_s > 0.0:
        args.a0 = abs(args.id_speed_mm_s) / max(abs(args.vmax_mm_s), 1.0e-12)
    if args.final_id_source == 'zero_zeta':
        args.id_method = 'zero-zeta-ls'
    if not args.log_csv:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        profile_tag = args.profile.replace('-', '_')
        axis_tag = args.axis.lower()
        target_tag = int(round(args.target_distance_mm))
        speed_tag = int(round(abs(args.vmax_mm_s)))
        output_dir = Path(args.run_output_dir).expanduser()
        args.log_csv = str(
            output_dir
            / f'{timestamp}_{profile_tag}_{axis_tag}{target_tag}_v{speed_tag}.csv'
        )
    return args


def check_args(args) -> bool:
    if args.target_distance_mm <= 0.0:
        print('Refusing: --target-distance-mm must be positive', file=sys.stderr)
        return False
    if abs(args.vmax_mm_s) < 1.0e-9:
        print('Refusing: --vmax-mm-s must be nonzero', file=sys.stderr)
        return False
    if not math.isfinite(args.pulse_pre_delay_s) or args.pulse_pre_delay_s < 0.0:
        print('Refusing: --pulse-pre-delay-s must be finite and nonnegative', file=sys.stderr)
        return False
    if not (0.0 < args.a0 < 1.0):
        print('Refusing: --a0 must be in (0, 1)', file=sys.stderr)
        return False
    if not math.isfinite(args.tau) or args.tau <= 0.0:
        print('Refusing: --tau must be finite and positive', file=sys.stderr)
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
    if not math.isfinite(args.gantry_fresh_timeout) or args.gantry_fresh_timeout <= 0.0:
        print('Refusing: --gantry-fresh-timeout must be finite and positive', file=sys.stderr)
        return False
    if args.integral_id_window_s < 0.0:
        print('Refusing: --integral-id-window-s must be nonnegative', file=sys.stderr)
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
    if args.is2_stable_max_range_s < 0.0:
        print('Refusing: --is2-stable-max-range-s must be nonnegative', file=sys.stderr)
        return False
    if args.is2_id_t_min < 0.0 or args.is2_id_t_max < 0.0:
        print('Refusing: IS2 ID_T gates must be nonnegative', file=sys.stderr)
        return False
    if args.is2_id_t_max > 0.0 and args.is2_id_t_max <= args.is2_id_t_min:
        print('Refusing: --is2-id-t-max must be greater than --is2-id-t-min', file=sys.stderr)
        return False
    if args.is2_schedule_margin_s < 0.0:
        print('Refusing: --is2-schedule-margin-s must be nonnegative', file=sys.stderr)
        return False
    if args.mode_fit_t_min <= 0.0 or args.mode_fit_t_max <= args.mode_fit_t_min:
        print('Refusing: mode-fit T range must satisfy 0 < min < max', file=sys.stderr)
        return False
    if args.mode_fit_window_s <= 0.0 or args.mode_fit_history_s <= 0.0:
        print('Refusing: mode-fit windows must be positive', file=sys.stderr)
        return False
    if args.mode_fit_min_samples < 4 or args.mode_fit_grid_count < 5:
        print('Refusing: mode-fit needs at least 4 samples and 5 grid points', file=sys.stderr)
        return False
    if args.zero_zeta_history_s <= 0.0:
        print('Refusing: --zero-zeta-history-s must be positive', file=sys.stderr)
        return False
    if args.zero_zeta_window_s < 0.0:
        print('Refusing: --zero-zeta-window-s must be nonnegative', file=sys.stderr)
        return False
    if args.zero_zeta_update_period_s <= 0.0:
        print('Refusing: --zero-zeta-update-period-s must be positive', file=sys.stderr)
        return False
    if args.zero_zeta_t_min <= 0.0 or args.zero_zeta_t_max <= args.zero_zeta_t_min:
        print('Refusing: zero-zeta T range must satisfy 0 < min < max', file=sys.stderr)
        return False
    if args.zero_zeta_min_samples < 4 or args.zero_zeta_grid_count < 5:
        print('Refusing: zero-zeta needs at least 4 samples and 5 grid points', file=sys.stderr)
        return False
    if args.zero_zeta_min_p2p_mm < 0.0 or args.zero_zeta_min_amp_mm < 0.0:
        print('Refusing: zero-zeta amplitude gates must be nonnegative', file=sys.stderr)
        return False
    if args.zero_zeta_max_norm_rmse <= 0.0:
        print('Refusing: --zero-zeta-max-norm-rmse must be positive', file=sys.stderr)
        return False
    if args.freq_bank_update_period_s <= 0.0:
        print('Refusing: --freq-bank-update-period-s must be positive', file=sys.stderr)
        return False
    if args.freq_bank_min_ratio < 1.0:
        print('Refusing: --freq-bank-min-ratio must be at least 1.0', file=sys.stderr)
        return False
    if args.two_mode_window_s <= 0.0 or args.two_mode_history_s <= 0.0:
        print('Refusing: two-mode windows must be positive', file=sys.stderr)
        return False
    if args.two_mode_t1_min <= 0.0 or args.two_mode_t1_max <= args.two_mode_t1_min:
        print('Refusing: two-mode T1 range must satisfy 0 < min < max', file=sys.stderr)
        return False
    if args.two_mode_t2_min <= 0.0 or args.two_mode_t2_max <= args.two_mode_t2_min:
        print('Refusing: two-mode T2 range must satisfy 0 < min < max', file=sys.stderr)
        return False
    if args.two_mode_t1_grid_count < 5 or args.two_mode_t2_grid_count < 5:
        print('Refusing: two-mode grid counts must be at least 5', file=sys.stderr)
        return False
    if args.two_mode_min_samples < 8:
        print('Refusing: --two-mode-min-samples must be at least 8', file=sys.stderr)
        return False
    if args.two_mode_update_period_s < 0.0:
        print('Refusing: --two-mode-update-period-s must be nonnegative', file=sys.stderr)
        return False
    if args.two_mode_max_t2_t1_ratio <= 0.0:
        print('Refusing: --two-mode-max-t2-t1-ratio must be positive', file=sys.stderr)
        return False
    if args.paper2_window_s <= 0.0 or args.paper2_history_s <= 0.0:
        print('Refusing: Paper 2 ID windows must be positive', file=sys.stderr)
        return False
    if args.paper2_min_samples < 3 or args.paper2_min_extrema < 2:
        print('Refusing: Paper 2 ID needs at least 3 samples and 2 extrema', file=sys.stderr)
        return False
    if args.paper2_min_peak_dt <= 0.0:
        print('Refusing: --paper2-min-peak-dt must be positive', file=sys.stderr)
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
    if not math.isfinite(args.nonzero_ic_omega_rad_s):
        print('Refusing: --nonzero-ic-omega-rad-s must be finite', file=sys.stderr)
        return False
    if (
        not math.isfinite(args.nonzero_ic_shaper_frequency_scale)
        or args.nonzero_ic_shaper_frequency_scale < 0.8
        or args.nonzero_ic_shaper_frequency_scale > 1.2
    ):
        print(
            'Refusing: --nonzero-ic-shaper-frequency-scale must be finite '
            'and in [0.8, 1.2]',
            file=sys.stderr,
        )
        return False
    if (
        args.profile != 'nonzero-ic'
        and not math.isclose(
            args.nonzero_ic_shaper_frequency_scale,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        print(
            'Refusing: --nonzero-ic-shaper-frequency-scale only applies to '
            '--profile nonzero-ic',
            file=sys.stderr,
        )
        return False
    if (
        args.nonzero_ic_finite_amplitude_correction
        and args.profile != 'nonzero-ic'
    ):
        print(
            'Refusing: --nonzero-ic-finite-amplitude-correction only applies '
            'to --profile nonzero-ic',
            file=sys.stderr,
        )
        return False
    if args.nonzero_ic_robust and args.profile != 'nonzero-ic':
        print(
            'Refusing: --nonzero-ic-robust only applies to '
            '--profile nonzero-ic',
            file=sys.stderr,
        )
        return False
    if (
        not math.isfinite(args.nonzero_ic_robust_band_fraction)
        or args.nonzero_ic_robust_band_fraction <= 0.0
        or args.nonzero_ic_robust_band_fraction >= 0.25
    ):
        print(
            'Refusing: --nonzero-ic-robust-band-fraction must be finite '
            'and in (0, 0.25)',
            file=sys.stderr,
        )
        return False
    if (
        args.nonzero_ic_finite_amplitude_correction
        and not args.nonzero_ic_adaptive_frequency
    ):
        print(
            'Refusing: --nonzero-ic-finite-amplitude-correction requires '
            '--nonzero-ic-adaptive-frequency',
            file=sys.stderr,
        )
        return False
    if (
        not math.isfinite(args.nonzero_ic_shaper_omega_rad_s)
        or args.nonzero_ic_shaper_omega_rad_s < 0.0
    ):
        print(
            'Refusing: --nonzero-ic-shaper-omega-rad-s must be finite and '
            'nonnegative',
            file=sys.stderr,
        )
        return False
    if (
        args.nonzero_ic_finite_amplitude_correction
        and args.nonzero_ic_shaper_omega_rad_s > 0.0
    ):
        print(
            'Refusing: fixed --nonzero-ic-shaper-omega-rad-s bypasses the '
            'finite-amplitude correction; choose only one',
            file=sys.stderr,
        )
        return False
    if (
        not math.isfinite(args.nonzero_ic_omega_min_rad_s)
        or not math.isfinite(args.nonzero_ic_omega_max_rad_s)
        or args.nonzero_ic_omega_min_rad_s <= 0.0
        or args.nonzero_ic_omega_max_rad_s
        <= args.nonzero_ic_omega_min_rad_s
    ):
        print(
            'Refusing: adaptive omega bounds must satisfy 0 < min < max',
            file=sys.stderr,
        )
        return False
    if args.nonzero_ic_shaper_omega_rad_s > 0.0:
        if args.profile != 'nonzero-ic':
            print(
                'Refusing: --nonzero-ic-shaper-omega-rad-s only applies to '
                '--profile nonzero-ic',
                file=sys.stderr,
            )
            return False
        if not math.isclose(
            args.nonzero_ic_shaper_frequency_scale,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            print(
                'Refusing: use either --nonzero-ic-shaper-omega-rad-s or '
                '--nonzero-ic-shaper-frequency-scale, not both',
                file=sys.stderr,
            )
            return False
        if not (
            args.nonzero_ic_omega_min_rad_s
            <= args.nonzero_ic_shaper_omega_rad_s
            <= args.nonzero_ic_omega_max_rad_s
        ):
            print(
                'Refusing: --nonzero-ic-shaper-omega-rad-s must lie inside '
                'the configured adaptive omega search bounds',
                file=sys.stderr,
            )
            return False
    if (
        args.nonzero_ic_frequency_grid_count < 5
        or args.nonzero_ic_min_fit_samples < 6
    ):
        print(
            'Refusing: adaptive nonzero-IC fit needs at least 5 frequency '
            'grid points and 6 samples',
            file=sys.stderr,
        )
        return False
    if (
        not math.isfinite(args.nonzero_ic_min_fit_amplitude_mm)
        or args.nonzero_ic_min_fit_amplitude_mm <= 0.0
        or not math.isfinite(args.nonzero_ic_max_fit_nrmse)
        or args.nonzero_ic_max_fit_nrmse <= 0.0
        or not math.isfinite(args.nonzero_ic_fit_update_period_s)
        or args.nonzero_ic_fit_update_period_s <= 0.0
        or not math.isfinite(
            args.nonzero_ic_peak_velocity_tolerance_mm_s
        )
        or args.nonzero_ic_peak_velocity_tolerance_mm_s <= 0.0
    ):
        print(
            'Refusing: adaptive nonzero-IC fit/peak gates and update period must '
            'be finite and positive',
            file=sys.stderr,
        )
        return False
    if (
        args.profile == 'nonzero-ic'
        and args.nonzero_ic_adaptive_frequency
        and args.tau < 1.0
    ):
        print(
            'Refusing: adaptive nonzero-IC frequency fitting needs --tau >= 1.0s',
            file=sys.stderr,
        )
        return False
    if (
        not math.isfinite(args.timed_profile_lead_s)
        or args.timed_profile_lead_s < 0.15
        or args.timed_profile_lead_s > 2.0
    ):
        print(
            'Refusing: --timed-profile-lead-s must be finite and in [0.15, 2.0]s',
            file=sys.stderr,
        )
        return False
    if (
        args.profile == 'nonzero-ic'
        and args.nonzero_ic_robust
        and args.controller_timed_profile
        and args.timed_profile_lead_s < 0.75
    ):
        print(
            'Refusing: robust nonzero-IC optimization with a controller-timed '
            'profile needs --timed-profile-lead-s >= 0.75s',
            file=sys.stderr,
        )
        return False
    if (
        not math.isfinite(args.timed_profile_actuation_lead_ms)
        or args.timed_profile_actuation_lead_ms < 0.0
        or args.timed_profile_actuation_lead_ms > 100.0
    ):
        print(
            'Refusing: --timed-profile-actuation-lead-ms must be finite and '
            'in [0, 100]ms',
            file=sys.stderr,
        )
        return False
    if (
        not math.isfinite(args.nonzero_ic_max_payload_queue_delay_ms)
        or args.nonzero_ic_max_payload_queue_delay_ms < 0.0
        or args.nonzero_ic_max_payload_queue_delay_ms > 100.0
    ):
        print(
            'Refusing: --nonzero-ic-max-payload-queue-delay-ms must be '
            'finite and in [0, 100]ms',
            file=sys.stderr,
        )
        return False
    if (
        args.timed_profile_actuation_lead_ms > 0.0
        and (
            args.profile != 'nonzero-ic'
            or not args.controller_timed_profile
        )
    ):
        print(
            'Refusing: --timed-profile-actuation-lead-ms requires '
            '--profile nonzero-ic with --controller-timed-profile',
            file=sys.stderr,
        )
        return False
    if (
        args.profile == 'nonzero-ic'
        and args.controller_timed_profile
        and not args.nonzero_ic_adaptive_frequency
    ):
        print(
            'Refusing: --controller-timed-profile requires adaptive frequency/state fitting',
            file=sys.stderr,
        )
        return False
    if (
        not math.isfinite(args.nonzero_ic_max_command_speed_mm_s)
        or args.nonzero_ic_max_command_speed_mm_s <= 0.0
    ):
        print(
            'Refusing: --nonzero-ic-max-command-speed-mm-s must be finite and positive',
            file=sys.stderr,
        )
        return False
    if (
        args.profile == 'nonzero-ic'
        and args.nonzero_ic_max_command_speed_mm_s < abs(args.vmax_mm_s)
    ):
        print(
            'Refusing: --nonzero-ic-max-command-speed-mm-s must be at least '
            'abs(--vmax-mm-s)',
            file=sys.stderr,
        )
        return False
    if args.excite_only and not args.excite:
        print('Refusing: --excite-only requires --excite', file=sys.stderr)
        return False
    if args.excite:
        if args.profile != 'nonzero-ic':
            print(
                'Refusing: --excite is only supported with --profile nonzero-ic',
                file=sys.stderr,
            )
            return False
        excite_finite = (
            args.excite_target_angle_deg,
            args.excite_angle_tolerance_deg,
            args.excite_angle_band_deg,
            args.excite_speed_mm_s,
            args.excite_travel_budget_mm,
            args.excite_initial_excursion_mm,
            args.excite_excursion_step_mm,
            args.excite_position_kp_s,
            args.excite_return_speed_mm_s,
            args.excite_return_tolerance_mm,
            args.excite_slew_mm_s2,
            args.excite_timeout_s,
            args.excite_settle_cycles,
            args.excite_standclear_s,
            args.excite_abort_angle_deg,
            args.excite_omega_rad_s,
        )
        if any(not math.isfinite(value) for value in excite_finite):
            print('Refusing: --excite-* values must be finite', file=sys.stderr)
            return False
        if not (0.0 < args.excite_target_angle_deg < args.excite_abort_angle_deg):
            print(
                'Refusing: need 0 < --excite-target-angle-deg < '
                '--excite-abort-angle-deg',
                file=sys.stderr,
            )
            return False
        if args.excite_angle_tolerance_deg <= 0.0:
            print('Refusing: --excite-angle-tolerance-deg must be positive', file=sys.stderr)
            return False
        if args.excite_angle_band_deg < args.excite_angle_tolerance_deg:
            print(
                'Refusing: --excite-angle-band-deg must be >= '
                '--excite-angle-tolerance-deg',
                file=sys.stderr,
            )
            return False
        if not (0.0 < args.excite_speed_mm_s <= abs(args.vmax_mm_s)):
            print(
                'Refusing: --excite-speed-mm-s must be in (0, abs(--vmax-mm-s)]',
                file=sys.stderr,
            )
            return False
        if args.excite_travel_budget_mm <= 0.0 or args.excite_timeout_s <= 0.0:
            print(
                'Refusing: --excite-travel-budget-mm and --excite-timeout-s must '
                'be positive',
                file=sys.stderr,
            )
            return False
        if not (
            0.0 < args.excite_initial_excursion_mm
            <= args.excite_travel_budget_mm
            and args.excite_excursion_step_mm > 0.0
        ):
            print(
                'Refusing: need 0 < --excite-initial-excursion-mm <= '
                '--excite-travel-budget-mm and a positive excursion step',
                file=sys.stderr,
            )
            return False
        if not (
            args.excite_position_kp_s > 0.0
            and args.excite_return_speed_mm_s > 0.0
            and args.excite_return_tolerance_mm > 0.0
            and args.excite_slew_mm_s2 > 0.0
        ):
            print(
                'Refusing: bounded excitation feedback, return, tolerance, and '
                'slew parameters must be positive',
                file=sys.stderr,
            )
            return False
        if args.excite_settle_cycles <= 0.0 or args.excite_standclear_s < 0.0:
            print(
                'Refusing: --excite-settle-cycles must be positive and '
                '--excite-standclear-s non-negative',
                file=sys.stderr,
            )
            return False
        if args.excite_omega_rad_s < 0.0:
            print('Refusing: --excite-omega-rad-s must be >= 0', file=sys.stderr)
            return False

    workspace_values = (
        args.workspace_min_mm,
        args.workspace_max_mm,
        args.workspace_margin_mm,
    )
    if any(not math.isfinite(value) for value in workspace_values):
        print('Refusing: workspace limits must be finite', file=sys.stderr)
        return False
    if (
        args.workspace_margin_mm < 0.0
        or args.workspace_max_mm
        <= args.workspace_min_mm + 2.0 * args.workspace_margin_mm
    ):
        print('Refusing: invalid workspace bounds or margin', file=sys.stderr)
        return False
    for option_name, value in (
        ('--initial-swing-mm', args.initial_swing_mm),
        ('--initial-payload-velocity-mm-s', args.initial_payload_velocity_mm_s),
    ):
        if value is not None and not math.isfinite(value):
            print(f'Refusing: {option_name} must be finite', file=sys.stderr)
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
    # A pulse baseline is a single constant-velocity interval followed by
    # zero.  It has no delayed shaper switches, so the T/2T feasibility guards
    # below do not apply; only the general speed, travel, and workspace checks
    # above are required.
    if args.profile == 'pulse':
        return True

    if args.profile in ('robust', 'zv'):
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
    elif args.profile == 'nonzero-ic':
        if (
            not args.nonzero_ic_adaptive_frequency
            and args.initial_swing_mm is not None
            and args.initial_payload_velocity_mm_s is not None
        ):
            omega_n = (
                args.nonzero_ic_omega_rad_s
                if args.nonzero_ic_omega_rad_s > 0.0
                else math.sqrt(args.gravity_m_s2 / args.robust_rope_length_m)
            )
            direction = 1.0 if args.vmax_mm_s >= 0.0 else -1.0
            try:
                solve_nonzero_ic_shaper(
                    initial_swing_mm=direction * args.initial_swing_mm,
                    initial_payload_velocity_mm_s=(
                        direction * args.initial_payload_velocity_mm_s
                    ),
                    maximum_speed_mm_s=abs(args.vmax_mm_s),
                    omega_n_rad_s=omega_n,
                    move_duration_s=tf,
                    maximum_absolute_gain=(
                        args.nonzero_ic_max_command_speed_mm_s
                        / abs(args.vmax_mm_s)
                    ),
                )
            except ValueError as exc:
                print(f'Refusing: invalid nonzero-IC profile: {exc}', file=sys.stderr)
                return False
    elif args.profile == 'zv':
        if tf <= T_guard:
            print(
                f'Refusing: ZV shaper needs tf=distance/vmax > T. '
                f'Here tf={tf:.3f}s and guard T={T_guard:.3f}s. '
                'Use lower --vmax-mm-s or larger --target-distance-mm.',
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
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
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
