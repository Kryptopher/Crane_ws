#!/usr/bin/env python3
"""
adaptive_shaper_player.py

One-axis adaptive input-shaping test for the gantry crane.

This node runs TRAJ realtime STREAM mode:
  1. Commands an unshaped constant-velocity ID/cruise move.
  2. Runs the integral estimator online from /payload/pose_e_rel.
  3. Accepts the first stable, well-conditioned estimate.
  4. Starts the shaped stop as soon as the estimate is accepted.

The first hardware target is conservative: prove online ID can update the
shaper used by the active move. The start is intentionally unshaped so the
payload is excited; the stop is shaped using the online estimate.
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
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import SetMode

from adaptive_id_player import AdaptiveIdentifier, Estimate


@dataclass
class AcceptedShaper:
    accepted_at: float
    T: float
    A0: float
    A1: float
    A2: float
    omega_n: float
    zeta: float
    cond_b: float


class AdaptiveShaperPlayer(Node):
    def __init__(self, args):
        super().__init__('adaptive_shaper_player')
        self.args = args

        self.latest_gantry_state: GantryState | None = None
        self.latest_payload_abs_mm: float | None = None
        self.latest_swing_mm: float | None = None
        self.latest_cart_q_mm: float | None = None
        self.latest_payload_time: float | None = None
        self.latest_payload_wall: float | None = None

        self.motion_started = False
        self.motion_t0_payload: float | None = None
        self.wall_start: float | None = None
        self.stop_start_wall: float | None = None
        self.final_zero_wall: float | None = None
        self.start_cart_q_mm: float | None = None
        self.stop_base_vx_mm_s = 0.0
        self.stop_base_vy_mm_s = 0.0
        self.done = False
        self.start_requested = False

        self.accepted: AcceptedShaper | None = None
        self.valid_estimates: list[Estimate] = []
        self.residual_samples: list[tuple[float, float]] = []
        self.csv_file = None
        self.csv_writer: csv.writer | None = None

        self.axis_velocity_mm_s = (
            float(args.vx_mm_s) if args.axis == 'x' else float(args.vy_mm_s)
        )
        self.identifier = AdaptiveIdentifier(
            K_mm_s=self.axis_velocity_mm_s,
            A0=args.a0,
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
        self._last_wait_ready_print = 0.0
        self._log_flush_count = 0

        self.get_logger().info('adaptive_shaper_player started')
        self.get_logger().info(
            f'Axis={args.axis} v=({args.vx_mm_s:.1f}, {args.vy_mm_s:.1f}) mm/s '
            f'min_id={args.min_id_duration:.2f}s max_id={args.id_duration:.2f}s '
            f'stream_rate={args.stream_rate_hz:.1f}Hz')
        if args.boost_duration > 0.0:
            self.get_logger().info(
                f'Boost ID: {args.boost_duration:.2f}s at initial speed, '
                f'then {args.cruise_vx_mm_s:.1f}/{args.cruise_vy_mm_s:.1f} mm/s')
        if args.target_distance_mm > 0.0:
            self.get_logger().info(
                f'Target move: {args.target_distance_mm:.1f} mm along {args.axis}')
        self.get_logger().info(
            f'Accept: condB<{args.accept_cond:.1f}, {args.accept_valid_count} valid estimates, '
            f'T in [{args.zv_t_min:.2f}, {args.zv_t_max:.2f}]s')
        self.get_logger().info(
            f'ID damping gate: zeta in [{args.id_zeta_min:.3f}, {args.id_zeta_max:.3f}], '
            f'shaper damping clamp: [{args.zeta_min:.3f}, {args.zeta_max:.3f}]')

    def destroy_node(self):
        self._publish_stream(0.0, 0.0)
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
            'id_time_sec',
            'cmd_vx_mm_s',
            'cmd_vy_mm_s',
            'payload_abs_mm',
            'swing_mm',
            'cart_q_mm',
            'target_traveled_mm',
            'target_remaining_mm',
            'stop_distance_mm',
            'cond_b',
            'omega_n_rad_s',
            'zeta',
            'T_sec',
            'A0',
            'A1',
            'A2',
            'accepted',
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
        age = time.monotonic() - self.latest_payload_wall
        return age <= self.args.payload_fresh_timeout

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
        if msg.command != TrajCmd.MOTION_START:
            return
        if self.motion_started:
            return
        self.motion_started = True
        self.wall_start = time.monotonic()
        if self.latest_payload_abs_mm is not None and self.latest_payload_time is not None:
            self._begin_id()
        self.get_logger().info('MOTION_START: adaptive move started')

    def _payload_cb(self, msg: Float64MultiArray):
        # /payload/pose_e_rel:
        # [time, pitch_deg, roll_deg, x_rel_m, y_rel_m, z_rel_m, vx, vy, vz]
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
            self._maybe_accept_estimate(est)

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
        self.motion_t0_payload = self.latest_payload_time
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

    def _maybe_accept_estimate(self, est: Estimate):
        if self.accepted is not None:
            return
        if self.stop_start_wall is not None:
            return
        if est.cond_b > self.args.accept_cond:
            return
        if not (self.args.zv_t_min <= est.T <= self.args.zv_t_max):
            return
        if not (self.args.id_zeta_min <= est.zeta <= self.args.id_zeta_max):
            return

        self.valid_estimates.append(est)
        max_count = max(self.args.accept_valid_count, self.args.stability_count)
        self.valid_estimates = self.valid_estimates[-max_count:]
        if len(self.valid_estimates) < self.args.accept_valid_count:
            return

        recent = self.valid_estimates[-self.args.stability_count:]
        if len(recent) >= 2:
            t_values = [e.T for e in recent]
            if max(t_values) - min(t_values) > self.args.stability_tol:
                return

        shaper_zeta = max(self.args.zeta_min, min(est.zeta, self.args.zeta_max))
        A0, A1, A2 = self._shaper_amplitudes(shaper_zeta)
        self.accepted = AcceptedShaper(
            accepted_at=est.t,
            T=est.T,
            A0=A0,
            A1=A1,
            A2=A2,
            omega_n=est.omega_n,
            zeta=est.zeta,
            cond_b=est.cond_b,
        )
        self.get_logger().info(
            'ACCEPTED adaptive shaper: '
            f't={est.t:.3f}s T={est.T:.4f}s id_zeta={est.zeta:.4f} '
            f'shaper_zeta={shaper_zeta:.4f} condB={est.cond_b:.3e} '
            f'A=[{A0:.4f},{A1:.4f},{A2:.4f}]')

    def _stream_timer_cb(self):
        if self.done:
            return
        if not self.start_requested and not self.motion_started:
            return
        now = time.monotonic()
        if self.wall_start is None:
            # A zero STREAM primes TRAJ realtime. If motors are already enabled,
            # the controller will publish MOTION_START and begin its clock.
            self._publish_stream(0.0, 0.0)
            return

        move_t = now - self.wall_start
        vx, vy = self._command_for_time(move_t)
        self._publish_stream(vx, vy)
        self._log_row(move_t, vx, vy)

        if self.stop_start_wall is not None:
            stop_age = now - self.stop_start_wall
            shaper = self.accepted
            tail = 2.0 * (shaper.T if shaper is not None else self.args.fallback_t)
            if stop_age >= tail and self.final_zero_wall is None:
                self.final_zero_wall = now
                self._publish_stream(0.0, 0.0)
                self.get_logger().info(
                    f'Shaped command is zero; collecting residual swing for '
                    f'{self.args.residual_window:.1f}s')
                return
            if self.final_zero_wall is not None:
                self._publish_stream(0.0, 0.0)
            if self.final_zero_wall is not None and now - self.final_zero_wall >= self.args.residual_window:
                self.done = True
                self._print_final_report()

    def _command_for_time(self, move_t: float) -> tuple[float, float]:
        accepted_ready = (
            self.accepted is not None
            and move_t >= self.args.min_id_duration
            and not self.args.continue_until_id_duration
        )
        target_stop_ready = (
            self.args.target_distance_mm > 0.0
            and accepted_ready
            and self._remaining_target_distance_mm() <= self._adaptive_stop_distance_mm()
        )
        fallback_target_stop_ready = (
            self.args.target_distance_mm > 0.0
            and self.accepted is None
            and self._remaining_target_distance_mm() <= self._adaptive_stop_distance_mm()
        )
        immediate_stop_ready = (
            accepted_ready
            and self.args.target_distance_mm <= 0.0
        )
        timed_out = (
            move_t >= self.args.id_duration
            and (self.accepted is None or self.args.target_distance_mm <= 0.0)
        )

        if self.stop_start_wall is None and (
            immediate_stop_ready or target_stop_ready or fallback_target_stop_ready or timed_out
        ):
            current_vx, current_vy = self._id_velocity_for_time(move_t)
            self._start_shaped_stop(current_vx, current_vy)
            if target_stop_ready:
                remaining = self._remaining_target_distance_mm()
                stop_dist = self._adaptive_stop_distance_mm()
                self.get_logger().info(
                    f'Target stop point reached at move_t={move_t:.2f}s '
                    f'remaining={remaining:.1f}mm stop_dist={stop_dist:.1f}mm')
            elif immediate_stop_ready:
                self.get_logger().info(
                    f'Accepted estimate ready at move_t={move_t:.2f}s; starting shaped stop')
            elif fallback_target_stop_ready:
                remaining = self._remaining_target_distance_mm()
                stop_dist = self._adaptive_stop_distance_mm()
                self.get_logger().warn(
                    f'No accepted estimate before target stop point at move_t={move_t:.2f}s; '
                    f'using fallback T={self.args.fallback_t:.3f}s '
                    f'remaining={remaining:.1f}mm stop_dist={stop_dist:.1f}mm')
            elif self.accepted is None:
                self.get_logger().warn(
                    f'No accepted estimate by {move_t:.2f}s; using fallback T={self.args.fallback_t:.3f}s')
            else:
                self.get_logger().info(
                    f'Max ID time reached; starting shaped stop with T={self.accepted.T:.4f}s')

        if self.stop_start_wall is None:
            return self._id_velocity_for_time(move_t)

        stop_age = time.monotonic() - self.stop_start_wall
        gain = self._stop_gain(stop_age)
        return self.stop_base_vx_mm_s * gain, self.stop_base_vy_mm_s * gain

    def _start_shaped_stop(self, vx_mm_s: float, vy_mm_s: float):
        self.stop_start_wall = time.monotonic()
        self.stop_base_vx_mm_s = float(vx_mm_s)
        self.stop_base_vy_mm_s = float(vy_mm_s)

    def _stop_gain(self, stop_age: float) -> float:
        shaper = self.accepted
        if shaper is None:
            T = self.args.fallback_t
            A0, A1, A2 = self.args.a0, 0.5, 1.0 - self.args.a0 - 0.5
            if A2 < 0.0:
                A0, A1, A2 = 0.25, 0.50, 0.25
        else:
            T, A0, A1, A2 = shaper.T, shaper.A0, shaper.A1, shaper.A2

        if stop_age < 0.0:
            return 1.0
        if stop_age < T:
            return max(0.0, 1.0 - A0)
        if stop_age < 2.0 * T:
            return max(0.0, 1.0 - A0 - A1)
        return 0.0

    def _id_velocity_for_time(self, move_t: float) -> tuple[float, float]:
        if self.args.boost_duration <= 0.0:
            return self.args.vx_mm_s, self.args.vy_mm_s
        if move_t < self.args.boost_duration:
            return self.args.vx_mm_s, self.args.vy_mm_s
        return self.args.cruise_vx_mm_s, self.args.cruise_vy_mm_s

    def _target_direction(self) -> float:
        v = self.args.vx_mm_s if self.args.axis == 'x' else self.args.vy_mm_s
        return 1.0 if v >= 0.0 else -1.0

    def _traveled_target_distance_mm(self) -> float:
        if self.start_cart_q_mm is None or self.latest_cart_q_mm is None:
            return 0.0
        return self._target_direction() * (self.latest_cart_q_mm - self.start_cart_q_mm)

    def _remaining_target_distance_mm(self) -> float:
        if self.args.target_distance_mm <= 0.0:
            return 0.0
        return max(0.0, self.args.target_distance_mm - self._traveled_target_distance_mm())

    def _adaptive_stop_distance_mm(self) -> float:
        move_t = 0.0 if self.wall_start is None else time.monotonic() - self.wall_start
        vx, vy = self._id_velocity_for_time(move_t)
        speed = math.hypot(vx, vy)
        shaper = self.accepted
        if shaper is None:
            T, A0, A1 = self.args.fallback_t, self.args.a0, 0.5
        else:
            T, A0, A1 = shaper.T, shaper.A0, shaper.A1
        gain_1 = max(0.0, 1.0 - A0)
        gain_2 = max(0.0, 1.0 - A0 - A1)
        return speed * T * (gain_1 + gain_2)

    def _shaper_amplitudes(self, zeta: float) -> tuple[float, float, float]:
        z = max(0.0, min(float(zeta), 0.99))
        wd_factor = math.sqrt(max(1.0 - z * z, 1.0e-12))
        exp1 = math.exp(math.pi * z / wd_factor)
        exp2 = math.exp(2.0 * math.pi * z / wd_factor)
        A0 = self.args.a0
        A1 = (A0 + (1.0 - A0) * exp2) / (exp1 + exp2)
        A2 = 1.0 - A0 - A1
        return A0, A1, A2

    def _print_final_report(self):
        traveled = self._traveled_target_distance_mm()
        target = self.args.target_distance_mm
        target_error = traveled - target if target > 0.0 else 0.0

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

        if self.accepted is None:
            estimate_text = 'estimate: fallback used'
        else:
            estimate_text = (
                f'estimate: T={self.accepted.T:.4f}s zeta={self.accepted.zeta:.4f} '
                f'condB={self.accepted.cond_b:.3e}'
            )

        if target > 0.0:
            travel_text = (
                f'travel: target={target:.1f} mm actual={traveled:.1f} mm '
                f'error={target_error:+.1f} mm'
            )
        else:
            travel_text = f'travel: actual={traveled:.1f} mm'

        self.get_logger().info(
            f'Adaptive shaped move complete | {travel_text} | '
            f'{sample_text} | {estimate_text}')

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
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds * 1.0e-9,
            move_t,
            '' if latest is None else latest.t,
            vx,
            vy,
            '' if self.latest_payload_abs_mm is None else self.latest_payload_abs_mm,
            '' if self.latest_swing_mm is None else self.latest_swing_mm,
            '' if self.latest_cart_q_mm is None else self.latest_cart_q_mm,
            self._traveled_target_distance_mm(),
            self._remaining_target_distance_mm(),
            self._adaptive_stop_distance_mm(),
            '' if latest is None else latest.cond_b,
            '' if latest is None else latest.omega_n,
            '' if latest is None else latest.zeta,
            '' if latest is None else latest.T,
            '' if latest is None else latest.A0,
            '' if latest is None else latest.A1,
            '' if latest is None else latest.A2,
            int(self.accepted is not None),
        ])
        if self.csv_file is not None:
            self._log_flush_count += 1
            if self._log_flush_count % 25 == 0:
                self.csv_file.flush()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis', default='x', choices=['x', 'y'])
    parser.add_argument('--payload-topic', default='/payload/pose_e_rel')
    parser.add_argument('--vx-mm-s', type=float, default=250.0)
    parser.add_argument('--vy-mm-s', type=float, default=0.0)
    parser.add_argument(
        '--boost-duration',
        type=float,
        default=0.0,
        help='Optional initial excitation duration at vx/vy before switching to cruise_vx/cruise_vy.')
    parser.add_argument('--cruise-vx-mm-s', type=float, default=0.0)
    parser.add_argument('--cruise-vy-mm-s', type=float, default=0.0)
    parser.add_argument('--id-duration', type=float, default=2.0)
    parser.add_argument(
        '--target-distance-mm',
        type=float,
        default=0.0,
        help='If >0, keep cruising after ID and shape the stop to finish this travel distance.')
    parser.add_argument(
        '--min-id-duration',
        type=float,
        default=0.5,
        help='Earliest allowed shaped stop after an accepted estimate. Default: 0.5s')
    parser.add_argument(
        '--continue-until-id-duration',
        action='store_true',
        help='Keep moving until --id-duration even if a valid estimate is accepted early.')
    parser.add_argument('--max-travel-mm', type=float, default=800.0)
    parser.add_argument('--stream-rate-hz', type=float, default=100.0)
    parser.add_argument('--payload-fresh-timeout', type=float, default=0.25)
    parser.add_argument('--a0', type=float, default=0.25)
    parser.add_argument('--cond-threshold', type=float, default=1.0e8)
    parser.add_argument('--accept-cond', type=float, default=1000.0)
    parser.add_argument('--accept-valid-count', type=int, default=3)
    parser.add_argument('--stability-count', type=int, default=3)
    parser.add_argument('--stability-tol', type=float, default=0.08)
    parser.add_argument('--omega-min', type=float, default=0.5)
    parser.add_argument('--omega-max', type=float, default=20.0)
    parser.add_argument('--id-zeta-min', type=float, default=0.0)
    parser.add_argument('--id-zeta-max', type=float, default=0.99)
    parser.add_argument('--zeta-min', type=float, default=0.0)
    parser.add_argument('--zeta-max', type=float, default=0.99)
    parser.add_argument('--zv-t-min', type=float, default=0.2)
    parser.add_argument('--zv-t-max', type=float, default=3.0)
    parser.add_argument('--fallback-t', type=float, default=1.0)
    parser.add_argument(
        '--residual-window',
        type=float,
        default=2.0,
        help='Seconds of zero command after shaped stop used for residual swing metrics.')
    parser.add_argument('--log-csv', default='')
    parser.add_argument('--no-arm', action='store_true')
    parser.add_argument('--no-auto-enable', action='store_true')
    parser.add_argument('--estimate-topic', default='/adaptive_shaper/estimate')
    return parser.parse_args()


def check_args(args) -> bool:
    if args.id_duration <= 0.0:
        print('Refusing: --id-duration must be positive', file=sys.stderr)
        return False
    if args.min_id_duration < 0.0 or args.min_id_duration > args.id_duration:
        print('Refusing: --min-id-duration must be between 0 and --id-duration', file=sys.stderr)
        return False
    if args.target_distance_mm < 0.0:
        print('Refusing: --target-distance-mm must be non-negative', file=sys.stderr)
        return False
    if args.stream_rate_hz <= 0.0:
        print('Refusing: --stream-rate-hz must be positive', file=sys.stderr)
        return False
    if args.payload_fresh_timeout <= 0.0:
        print('Refusing: --payload-fresh-timeout must be positive', file=sys.stderr)
        return False
    if args.axis == 'x' and abs(args.vx_mm_s) < 1.0e-9:
        print('Refusing: --axis x needs nonzero --vx-mm-s', file=sys.stderr)
        return False
    if args.axis == 'y' and abs(args.vy_mm_s) < 1.0e-9:
        print('Refusing: --axis y needs nonzero --vy-mm-s', file=sys.stderr)
        return False
    if abs(args.vx_mm_s) > 1.0e-9 and abs(args.vy_mm_s) > 1.0e-9:
        print('Refusing: first pass is one-axis only', file=sys.stderr)
        return False
    if args.boost_duration < 0.0:
        print('Refusing: --boost-duration must be non-negative', file=sys.stderr)
        return False
    if args.boost_duration > args.id_duration:
        print('Refusing: --boost-duration must be <= --id-duration', file=sys.stderr)
        return False
    if args.boost_duration > 0.0:
        if args.axis == 'x' and abs(args.cruise_vy_mm_s) > 1.0e-9:
            print('Refusing: x-axis boost test needs --cruise-vy-mm-s 0', file=sys.stderr)
            return False
        if args.axis == 'y' and abs(args.cruise_vx_mm_s) > 1.0e-9:
            print('Refusing: y-axis boost test needs --cruise-vx-mm-s 0', file=sys.stderr)
            return False
        if args.axis == 'x' and abs(args.cruise_vx_mm_s) < 1.0e-9:
            args.cruise_vx_mm_s = args.vx_mm_s
        if args.axis == 'y' and abs(args.cruise_vy_mm_s) < 1.0e-9:
            args.cruise_vy_mm_s = args.vy_mm_s
    if not (0.0 < args.a0 < 1.0):
        print('Refusing: --a0 must be in (0, 1)', file=sys.stderr)
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
    if args.id_zeta_min < 0.0 or args.id_zeta_max <= args.id_zeta_min:
        print('Refusing: invalid --id-zeta-min/--id-zeta-max', file=sys.stderr)
        return False
    if args.zeta_min < 0.0 or args.zeta_max < args.zeta_min:
        print('Refusing: invalid --zeta-min/--zeta-max', file=sys.stderr)
        return False
    if args.fallback_t <= 0.0:
        print('Refusing: --fallback-t must be positive', file=sys.stderr)
        return False
    if args.residual_window <= 0.0:
        print('Refusing: --residual-window must be positive', file=sys.stderr)
        return False

    T_guard = min(args.zv_t_max, max(args.fallback_t, 1.2))
    # A shaped stop continues for 2*T, but not at full speed. For the normal
    # A=[0.25,0.50,0.25] case the extra stop distance is V*T. Use 1.25*T as a
    # small guard for the lightly damped adaptive coefficients.
    if args.boost_duration > 0.0:
        boost_v = math.hypot(args.vx_mm_s, args.vy_mm_s)
        cruise_v = math.hypot(args.cruise_vx_mm_s, args.cruise_vy_mm_s)
        id_travel = (
            boost_v * args.boost_duration
            + cruise_v * max(0.0, args.id_duration - args.boost_duration)
        )
        stop_v = cruise_v
    else:
        stop_v = math.hypot(args.vx_mm_s, args.vy_mm_s)
        id_travel = stop_v * args.id_duration
    if args.target_distance_mm > 0.0:
        travel = args.target_distance_mm
    else:
        travel = id_travel + stop_v * 1.25 * T_guard
    if travel > args.max_travel_mm:
        print(
            f'Refusing: guarded travel {travel:.1f} mm > {args.max_travel_mm:.1f} mm. '
            'Increase --max-travel-mm only if the start position has enough room.',
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
        node = AdaptiveShaperPlayer(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[adaptive_shaper_player] {exc}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
