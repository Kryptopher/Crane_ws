#!/usr/bin/env python3
"""
adaptive_step_is_player.py

Experimental step-reference adaptive input shaper based on the colleague's
simulation code:

  ref(t) = K                         for 0 <= t <= tau
         = K + A                     for tau+T <= t < tau+2T
         = 1                         for t >= tau+2T

The physical gantry only accepts velocity stream commands, so this node tracks
those reference levels with a saturated position servo in TRAJ realtime mode.
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
import sympy as sy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.signal import find_peaks

from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import SetMode


@dataclass
class StepEstimate:
    estimated_at: float
    zeta: float
    omega_n: float
    A: float
    T: float
    peak_time: float
    peak_value: float


class AdaptiveStepISPlayer(Node):
    def __init__(self, args):
        super().__init__('adaptive_step_is_player')
        self.args = args

        self.latest_gantry_state: GantryState | None = None
        self.latest_payload_time: float | None = None
        self.latest_payload_wall: float | None = None
        self.latest_payload_abs_mm: float | None = None
        self.latest_swing_mm: float | None = None
        self.latest_cart_q_mm: float | None = None

        self.motion_started = False
        self.start_requested = False
        self.wall_start: float | None = None
        self.start_cart_q_mm: float | None = None
        self.start_payload_abs_mm: float | None = None
        self.final_hold_wall: float | None = None
        self.done = False
        self.aborted = False

        self.estimate: StepEstimate | None = None
        self.id_samples: list[tuple[float, float]] = []
        self.residual_samples: list[tuple[float, float]] = []
        self.csv_file = None
        self.csv_writer: csv.writer | None = None
        self._log_flush_count = 0
        self._last_wait_ready_print = 0.0

        self.direction = 1.0 if args.target_distance_mm >= 0.0 else -1.0
        self.target_abs_mm = abs(float(args.target_distance_mm))

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.traj_pub = self.create_publisher(TrajCmd, '/traj_cmd', qos)
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

        self.get_logger().info('adaptive_step_is_player started')
        self.get_logger().info(
            f'axis={args.axis} target={args.target_distance_mm:.1f} mm K={args.K:.4f} '
            f'tau={args.tau:.3f}s track_vmax={args.track_vmax_mm_s:.1f} mm/s '
            f'kp={args.track_kp:.2f} 1/s')
        self.get_logger().info(
            f'ID step target={args.K * self.target_abs_mm:.1f} mm; '
            f'A/T gates: T=[{args.T_min:.2f},{args.T_max:.2f}]s zeta=[0,{args.zeta_max:.2f}]')

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
            'cmd_vx_mm_s',
            'cmd_vy_mm_s',
            'ref_frac',
            'ref_pos_mm',
            'payload_norm',
            'payload_abs_mm',
            'swing_mm',
            'cart_q_mm',
            'traveled_mm',
            'zeta',
            'omega_n_rad_s',
            'A',
            'T_sec',
            'phase',
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
        q = self.latest_cart_q_mm if self.latest_cart_q_mm is not None else 0.0
        x = self.latest_payload_abs_mm if self.latest_payload_abs_mm is not None else q
        self.start_cart_q_mm = q
        self.start_payload_abs_mm = x
        self.get_logger().info(
            f'MOTION_START: step IS move started q0={q:.2f}mm x0={x:.2f}mm')

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

        if self.final_hold_wall is not None:
            residual_t = self.latest_payload_wall - self.final_hold_wall
            if residual_t >= 0.0:
                self.residual_samples.append((residual_t, self.latest_swing_mm))

    def _current_cart_axis_m(self) -> float | None:
        if self.latest_gantry_state is None:
            return None
        return float(self.latest_gantry_state.x) if self.args.axis == 'x' else float(self.latest_gantry_state.y)

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
            self._log_row(move_t, 0.0, 0.0, 0.0, self.latest_cart_q_mm or 0.0, 'aborted')
            if self.final_hold_wall is not None and now - self.final_hold_wall >= self.args.residual_window:
                self.done = True
                self._print_final_report()
            return

        ref_frac, phase = self._reference_for_time(move_t)
        ref_pos_mm = self._reference_position_mm(ref_frac)
        vx_axis = self._track_reference_velocity(ref_pos_mm)
        vx, vy = (vx_axis, 0.0) if self.args.axis == 'x' else (0.0, vx_axis)

        if move_t <= self.args.tau and self.start_payload_abs_mm is not None:
            self.id_samples.append((move_t, self._payload_norm()))
        if self.estimate is None and move_t >= self.args.tau:
            self._try_estimate(move_t)
            if self.aborted:
                self._publish_stream(0.0, 0.0)
                self._log_row(move_t, 0.0, 0.0, 0.0, self.latest_cart_q_mm or 0.0, 'aborted')
                return

        self._publish_stream(vx, vy)
        self._log_row(move_t, vx, vy, ref_frac, ref_pos_mm, phase)

        final_ref_active = self.estimate is not None and move_t >= self.args.tau + 2.0 * self.estimate.T
        at_target = abs(ref_pos_mm - (self.latest_cart_q_mm or ref_pos_mm)) <= self.args.position_tolerance_mm
        if final_ref_active and at_target and self.final_hold_wall is None:
            self.final_hold_wall = now
            self.get_logger().info(
                f'Final reference reached; collecting residual swing for {self.args.residual_window:.1f}s')
        if self.final_hold_wall is not None:
            self._publish_stream(0.0 if at_target else vx, 0.0 if at_target else vy)
            if now - self.final_hold_wall >= self.args.residual_window:
                self.done = True
                self._print_final_report()

    def _reference_for_time(self, move_t: float) -> tuple[float, str]:
        if self.estimate is None:
            return self.args.K, 'id'
        T = self.estimate.T
        if move_t < self.args.tau + T:
            return self.args.K, 'hold_K'
        if move_t < self.args.tau + 2.0 * T:
            return self.args.K + self.estimate.A, 'hold_K_plus_A'
        return 1.0, 'final'

    def _reference_position_mm(self, ref_frac: float) -> float:
        q0 = self.start_cart_q_mm if self.start_cart_q_mm is not None else 0.0
        return q0 + self.direction * ref_frac * self.target_abs_mm

    def _track_reference_velocity(self, ref_pos_mm: float) -> float:
        if self.latest_cart_q_mm is None:
            return 0.0
        err = ref_pos_mm - self.latest_cart_q_mm
        if abs(err) <= self.args.position_tolerance_mm:
            return 0.0
        vx = self.args.track_kp * err
        vmax = self.args.track_vmax_mm_s
        return max(-vmax, min(vmax, vx))

    def _payload_norm(self) -> float:
        if self.latest_payload_abs_mm is None or self.start_payload_abs_mm is None:
            return 0.0
        return self.direction * (self.latest_payload_abs_mm - self.start_payload_abs_mm) / self.target_abs_mm

    def _try_estimate(self, move_t: float):
        if len(self.id_samples) < 10:
            self._abort('not enough ID samples')
            return
        t = np.array([p[0] for p in self.id_samples], dtype=float)
        y = np.array([p[1] for p in self.id_samples], dtype=float)
        zeta, omega_n, peak_time, peak_value = self._estimate_second_order(t, y, self.args.K)
        if zeta is None or omega_n is None:
            self._abort(
                'peak estimator failed '
                f'(samples={len(y)} y_min={float(np.min(y)):.4f} '
                f'y_max={float(np.max(y)):.4f} ref={self.args.K:.4f})')
            return
        if zeta >= self.args.zeta_disable:
            self._abort(f'estimated zeta={zeta:.3f}; IS disabled')
            return
        zeta_for_calc = max(0.0, min(zeta, self.args.zeta_max))
        result = self._calculate_A_T(omega_n, zeta_for_calc)
        if result is None:
            self._abort('A/T solve failed')
            return
        A, T = result
        if not (self.args.T_min <= T <= self.args.T_max):
            self._abort(f'solved T={T:.3f}s outside gate')
            return
        if not (0.0 < A < 1.0 - self.args.K):
            self._abort(f'solved A={A:.3f} outside (0, 1-K)')
            return
        self.estimate = StepEstimate(
            estimated_at=move_t,
            zeta=zeta,
            omega_n=omega_n,
            A=A,
            T=T,
            peak_time=peak_time,
            peak_value=peak_value,
        )
        self.get_logger().info(
            f'LOCKED step IS: t={move_t:.3f}s zeta={zeta:.4f} '
            f'omega_n={omega_n:.4f}rad/s A={A:.4f} T={T:.4f}s '
            f'levels=[{self.args.K:.4f},{self.args.K + A:.4f},1.0000] '
            f'switches=[{self.args.tau + T:.3f},{self.args.tau + 2*T:.3f}]s')

    def _estimate_second_order(
        self,
        t: np.ndarray,
        y: np.ndarray,
        ref_level: float,
    ) -> tuple[float | None, float | None, float, float]:
        peaks, _ = find_peaks(y)
        mins, _ = find_peaks(-y)
        if len(peaks) == 0:
            return None, None, float('nan'), float('nan')

        peak_values_all = y[peaks]
        meaningful = peak_values_all > ref_level * (1.0 + self.args.min_overshoot_frac)
        if not np.any(meaningful):
            max_i = int(np.argmax(peak_values_all))
            if peak_values_all[max_i] <= ref_level:
                return None, None, float(t[peaks[max_i]]), float(peak_values_all[max_i])
            meaningful_peak_indices = np.array([peaks[max_i]], dtype=int)
        else:
            meaningful_peak_indices = peaks[meaningful]

        peak_values = y[meaningful_peak_indices]
        peak_times = t[meaningful_peak_indices]
        if len(peak_values) >= 2 and abs((peak_values[1] - peak_values[0]) / peak_values[0]) <= 1.0e-4:
            min_times = t[mins]
            if len(min_times) < 2:
                return None, None, float(peak_times[0]), float(peak_values[0])
            pos_dt = float(np.mean(np.diff(peak_times)))
            neg_dt = float(np.mean(np.diff(min_times)))
            period = 0.5 * (pos_dt + neg_dt)
            return 0.0, 2.0 * math.pi / period, float(peak_times[0]), float(peak_values[0])

        Mp = (float(peak_values[0]) - ref_level) / ref_level
        if Mp <= 0.0 or not math.isfinite(Mp):
            return None, None, float(peak_times[0]), float(peak_values[0])
        zeta = -math.log(Mp) / math.sqrt(math.pi * math.pi + math.log(Mp) ** 2)
        if zeta >= 1.0:
            return zeta, None, float(peak_times[0]), float(peak_values[0])
        omega_n = math.pi / (float(peak_times[0]) * math.sqrt(max(1.0 - zeta * zeta, 1.0e-12)))
        return zeta, omega_n, float(peak_times[0]), float(peak_values[0])

    def _calculate_A_T(self, omega_n: float, zeta: float) -> tuple[float, float] | None:
        A_sym, T_sym = sy.symbols('A,T')
        K = self.args.K
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

    def _abort(self, reason: str):
        self.aborted = True
        self.final_hold_wall = time.monotonic()
        self._publish_stream(0.0, 0.0)
        self.get_logger().error(f'Adaptive step IS aborted: {reason}')

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

    def _traveled_target_distance_mm(self) -> float:
        if self.start_cart_q_mm is None or self.latest_cart_q_mm is None:
            return 0.0
        return self.direction * (self.latest_cart_q_mm - self.start_cart_q_mm)

    def _log_row(self, move_t: float, vx: float, vy: float, ref_frac: float, ref_pos_mm: float, phase: str):
        if self.csv_writer is None:
            return
        est = self.estimate
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds * 1.0e-9,
            move_t,
            vx,
            vy,
            ref_frac,
            ref_pos_mm,
            self._payload_norm(),
            '' if self.latest_payload_abs_mm is None else self.latest_payload_abs_mm,
            '' if self.latest_swing_mm is None else self.latest_swing_mm,
            '' if self.latest_cart_q_mm is None else self.latest_cart_q_mm,
            self._traveled_target_distance_mm(),
            '' if est is None else est.zeta,
            '' if est is None else est.omega_n,
            '' if est is None else est.A,
            '' if est is None else est.T,
            'aborted' if self.aborted else phase,
        ])
        if self.csv_file is not None:
            self._log_flush_count += 1
            if self._log_flush_count % 25 == 0:
                self.csv_file.flush()

    def _print_final_report(self):
        traveled = self._traveled_target_distance_mm()
        samples = [x for _, x in self.residual_samples]
        if samples:
            p2p = max(samples) - min(samples)
            max_abs = max(abs(x) for x in samples)
            rms = math.sqrt(sum(x * x for x in samples) / len(samples))
            residual = (
                f'residual swing over {self.args.residual_window:.1f}s: '
                f'p2p={p2p:.2f} mm max_abs={max_abs:.2f} mm rms={rms:.2f} mm '
                f'samples={len(samples)}'
            )
        else:
            residual = 'residual swing: no samples'
        if self.estimate is None:
            est_text = 'estimate: none'
        else:
            est_text = (
                f'estimate: zeta={self.estimate.zeta:.4f} '
                f'omega_n={self.estimate.omega_n:.4f} A={self.estimate.A:.4f} '
                f'T={self.estimate.T:.4f}s')
        status = 'Adaptive step IS aborted' if self.aborted else 'Adaptive step IS complete'
        self.get_logger().info(
            f'{status} | travel: target={self.target_abs_mm:.1f} mm '
            f'actual={traveled:.1f} mm error={traveled - self.target_abs_mm:+.1f} mm | '
            f'{residual} | {est_text}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis', default='x', choices=['x', 'y'])
    parser.add_argument('--payload-topic', default='/payload/pose_e_rel')
    parser.add_argument('--target-distance-mm', type=float, default=750.0)
    parser.add_argument('--K', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--track-vmax-mm-s', type=float, default=300.0)
    parser.add_argument('--track-kp', type=float, default=6.0)
    parser.add_argument('--position-tolerance-mm', type=float, default=2.0)
    parser.add_argument(
        '--min-overshoot-frac',
        type=float,
        default=0.05,
        help='Peak estimator ignores peaks below K*(1+this fraction).')
    parser.add_argument('--stream-rate-hz', type=float, default=100.0)
    parser.add_argument('--payload-fresh-timeout', type=float, default=0.25)
    parser.add_argument('--T-min', type=float, default=0.2)
    parser.add_argument('--T-max', type=float, default=2.0)
    parser.add_argument('--zeta-max', type=float, default=0.95)
    parser.add_argument('--zeta-disable', type=float, default=0.99)
    parser.add_argument('--residual-window', type=float, default=3.0)
    parser.add_argument('--max-travel-mm', type=float, default=800.0)
    parser.add_argument('--log-csv', default='')
    parser.add_argument('--no-arm', action='store_true')
    parser.add_argument('--no-auto-enable', action='store_true')
    return parser.parse_args()


def check_args(args) -> bool:
    if args.target_distance_mm == 0.0:
        print('Refusing: --target-distance-mm must be nonzero', file=sys.stderr)
        return False
    if abs(args.target_distance_mm) > args.max_travel_mm:
        print('Refusing: target distance exceeds --max-travel-mm', file=sys.stderr)
        return False
    if not (0.0 < args.K < 1.0):
        print('Refusing: --K must be in (0, 1)', file=sys.stderr)
        return False
    if args.tau <= 0.0:
        print('Refusing: --tau must be positive', file=sys.stderr)
        return False
    if args.track_vmax_mm_s <= 0.0 or args.track_kp <= 0.0:
        print('Refusing: tracking speed and kp must be positive', file=sys.stderr)
        return False
    if args.min_overshoot_frac < 0.0:
        print('Refusing: --min-overshoot-frac must be non-negative', file=sys.stderr)
        return False
    if args.T_min <= 0.0 or args.T_max <= args.T_min:
        print('Refusing: invalid --T-min/--T-max', file=sys.stderr)
        return False
    if args.zeta_max < 0.0 or args.zeta_disable <= args.zeta_max:
        print('Refusing: invalid zeta gates', file=sys.stderr)
        return False
    if args.residual_window <= 0.0:
        print('Refusing: --residual-window must be positive', file=sys.stderr)
        return False
    return True


def main():
    args = parse_args()
    if not check_args(args):
        return 1
    rclpy.init()
    node = None
    try:
        node = AdaptiveStepISPlayer(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[adaptive_step_is_player] {exc}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
