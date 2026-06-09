#!/usr/bin/env python3
"""Two-phase adaptive input-shaping test.

Phase 1: unshaped constant-velocity ID move, then shaped stop.
Phase 2: after a short settle, run a fully shaped rest-to-rest target move
using the identified switch time T.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import SetMode
from adaptive_id_player import AdaptiveIdentifier, Estimate


class TwoPhasePlayer(Node):
    def __init__(self, args):
        super().__init__('adaptive_two_phase_player')
        self.args = args
        self.state = 'preflight'
        self.state_wall = time.monotonic()
        self.motion_wall: float | None = None
        self.latest_state: GantryState | None = None
        self.latest_payload_time: float | None = None
        self.latest_payload_wall: float | None = None
        self.latest_payload_abs_mm: float | None = None
        self.latest_swing_mm: float | None = None
        self.latest_cart_q_mm: float | None = None
        self.start_cart_q_mm: float | None = None
        self.shaped_start_q_mm: float | None = None
        self.accepted: Estimate | None = None
        self.shaper = None
        self.residual_samples: list[tuple[float, float]] = []
        self.csv_file = None
        self.csv_writer = None
        self._last_wait_print = 0.0
        self._flush_count = 0

        id_v = args.id_vx_mm_s if args.axis == 'x' else args.id_vy_mm_s
        self.identifier = AdaptiveIdentifier(
            K_mm_s=id_v,
            A0=args.a0,
            cond_threshold=args.cond_threshold,
            omega_min=args.omega_min,
            omega_max=args.omega_max,
            zeta_min=args.id_zeta_min,
            zeta_max=args.id_zeta_max,
        )

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.traj_pub = self.create_publisher(TrajCmd, '/traj_cmd', qos)
        self.enable_cli = self.create_client(Trigger, '/gantry/enable')
        self.mode_cli = self.create_client(SetMode, '/gantry/set_mode')
        self.create_subscription(GantryState, '/gantry/state', self._state_cb, 10)
        self.create_subscription(TrajCmd, '/traj_cmd', self._traj_cb, qos)
        self.create_subscription(Float64MultiArray, args.payload_topic, self._payload_cb, qos_profile_sensor_data)

        if args.log_csv:
            self._open_log(args.log_csv)
        self._arm_traj()
        self.timer = self.create_timer(1.0 / args.stream_rate_hz, self._timer_cb)
        self.start_timer = self.create_timer(0.5, self._start_cb)

        self.get_logger().info(
            f'two-phase adaptive player: ID v=({args.id_vx_mm_s:.1f},{args.id_vy_mm_s:.1f}) '
            f'move v=({args.move_vx_mm_s:.1f},{args.move_vy_mm_s:.1f}) '
            f'target={args.target_distance_mm:.1f}mm')

    def destroy_node(self):
        self._stream(0.0, 0.0)
        if self.csv_file is not None:
            self.csv_file.close()
        super().destroy_node()

    def _open_log(self, path_arg: str):
        path = Path(path_arg).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = path.open('w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'wall_time_sec', 'state', 'state_time_sec', 'cmd_vx_mm_s', 'cmd_vy_mm_s',
            'cart_q_mm', 'swing_mm', 'phase_travel_mm', 'cond_b', 'T_sec', 'zeta',
        ])
        self.csv_file.flush()
        self.get_logger().info(f'CSV log: {path}')

    def _arm_traj(self):
        if not self.mode_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/gantry/set_mode not available')
        req = SetMode.Request()
        req.mode = 'TRAJ'
        future = self.mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('failed to set TRAJ mode')
        self.get_logger().info('Gantry set to TRAJ mode')

    def _start_cb(self):
        if not self._ready():
            now = time.monotonic()
            if now - self._last_wait_print > 1.0:
                self._last_wait_print = now
                self.get_logger().info('Waiting for fresh /payload/pose_e_rel and /gantry/state')
            return
        self.start_timer.cancel()
        self.state = 'waiting_motion'
        if self.enable_cli.wait_for_service(timeout_sec=2.0):
            future = self.enable_cli.call_async(Trigger.Request())
            future.add_done_callback(lambda f: self.get_logger().info('/gantry/enable returned'))
            self.get_logger().info('Requested /gantry/enable')
        else:
            self.get_logger().warn('/gantry/enable unavailable; press Enable')

    def _ready(self) -> bool:
        if self.latest_state is None or self.latest_payload_wall is None:
            return False
        return time.monotonic() - self.latest_payload_wall <= self.args.payload_fresh_timeout

    def _state_cb(self, msg: GantryState):
        self.latest_state = msg

    def _traj_cb(self, msg: TrajCmd):
        if msg.command != TrajCmd.MOTION_START or self.motion_wall is not None:
            return
        self.motion_wall = time.monotonic()
        self.state = 'id'
        self.state_wall = self.motion_wall
        self._begin_id()
        self.get_logger().info('MOTION_START: phase 1 ID started')

    def _payload_cb(self, msg: Float64MultiArray):
        idx = 3 if self.args.axis == 'x' else 4
        if len(msg.data) <= idx:
            return
        swing_m = float(msg.data[idx])
        cart_m = self._cart_axis_m()
        if cart_m is None:
            cart_m = 0.0
        self.latest_payload_time = float(msg.data[0])
        self.latest_payload_wall = time.monotonic()
        self.latest_swing_mm = swing_m * 1000.0
        self.latest_cart_q_mm = cart_m * 1000.0
        self.latest_payload_abs_mm = (cart_m + swing_m) * 1000.0
        if self.state == 'id' and self.identifier.t0 is None:
            self._begin_id()
        if self.state == 'residual':
            self.residual_samples.append((time.monotonic() - self.state_wall, self.latest_swing_mm))
        if self.state == 'id' and self.identifier.t0 is not None:
            est = self.identifier.update(self.latest_payload_time, self.latest_payload_abs_mm, self.latest_cart_q_mm)
            if est is not None:
                self._maybe_accept(est)

    def _cart_axis_m(self):
        if self.latest_state is None:
            return None
        return float(self.latest_state.x) if self.args.axis == 'x' else float(self.latest_state.y)

    def _begin_id(self):
        if self.latest_payload_time is None or self.latest_payload_abs_mm is None:
            return
        q = 0.0 if self.latest_cart_q_mm is None else self.latest_cart_q_mm
        if self.start_cart_q_mm is None:
            self.start_cart_q_mm = q
        self.identifier.start(self.latest_payload_time, self.latest_payload_abs_mm, q_now_mm=q, zero_at_start=True)
        self.get_logger().info(f'ID integrator started: x={self.latest_payload_abs_mm:.2f}mm q={q:.2f}mm')

    def _maybe_accept(self, est: Estimate):
        if self.accepted is not None:
            return
        if est.t < self.args.min_id_duration:
            return
        if est.cond_b > self.args.accept_cond:
            return
        if not (self.args.zv_t_min <= est.T <= self.args.zv_t_max):
            return
        self.accepted = est
        self.shaper = self._make_shaper(est.T, max(self.args.zeta_min, min(est.zeta, self.args.zeta_max)))
        self.get_logger().info(
            f'ACCEPTED ID: t={est.t:.3f}s T={est.T:.4f}s id_zeta={est.zeta:.4f} '
            f'condB={est.cond_b:.3e} A=[{self.shaper[1]:.4f},{self.shaper[2]:.4f},{self.shaper[3]:.4f}]')

    def _make_shaper(self, T: float, zeta: float):
        z = max(0.0, min(float(zeta), 0.99))
        wd = math.sqrt(max(1.0 - z * z, 1.0e-12))
        e1 = math.exp(math.pi * z / wd)
        e2 = math.exp(2.0 * math.pi * z / wd)
        A0 = self.args.a0
        A1 = (A0 + (1.0 - A0) * e2) / (e1 + e2)
        A2 = 1.0 - A0 - A1
        return (float(T), A0, A1, A2)

    def _timer_cb(self):
        if self.state == 'done':
            return
        if self.motion_wall is None:
            if self.state == 'waiting_motion':
                self._stream(0.0, 0.0)
            return

        vx, vy = self._command()
        self._stream(vx, vy)
        self._log(vx, vy)

    def _command(self):
        t = time.monotonic() - self.state_wall
        if self.state == 'id':
            if self.accepted is not None and t >= self.args.min_id_duration:
                self._enter_id_stop()
            elif t >= self.args.id_duration:
                if self.accepted is None:
                    self.shaper = self._make_shaper(self.args.fallback_t, 0.0)
                    self.get_logger().warn(f'ID timeout; using fallback T={self.args.fallback_t:.3f}s')
                self._enter_id_stop()
            else:
                return self.args.id_vx_mm_s, self.args.id_vy_mm_s

        if self.state == 'id_stop':
            return self._stop_command(self.args.id_vx_mm_s, self.args.id_vy_mm_s, t)

        if self.state == 'settle':
            if t >= self.args.settle_after_id:
                self.state = 'shape_move'
                self.state_wall = time.monotonic()
                self.shaped_start_q_mm = self.latest_cart_q_mm
                self.get_logger().info('Phase 2 fully shaped target move started')
            return 0.0, 0.0

        if self.state == 'shape_move':
            return self._shape_move_command(t)

        if self.state == 'residual':
            if t >= self.args.residual_window:
                self._print_report()
                self.state = 'done'
            return 0.0, 0.0

        return 0.0, 0.0

    def _enter_id_stop(self):
        self.state = 'id_stop'
        self.state_wall = time.monotonic()
        self.get_logger().info('Phase 1 ID complete; shaped stop to rest')

    def _stop_command(self, vx0, vy0, t):
        T, A0, A1, _ = self.shaper
        if t < T:
            g = max(0.0, 1.0 - A0)
        elif t < 2.0 * T:
            g = max(0.0, 1.0 - A0 - A1)
        else:
            self.state = 'settle'
            self.state_wall = time.monotonic()
            self.get_logger().info(f'Settling for {self.args.settle_after_id:.1f}s before shaped target')
            return 0.0, 0.0
        return vx0 * g, vy0 * g

    def _shape_move_command(self, t):
        T, A0, A1, _ = self.shaper
        vx = self.args.move_vx_mm_s
        vy = self.args.move_vy_mm_s
        speed = math.hypot(vx, vy)
        if speed <= 1.0e-9:
            return 0.0, 0.0
        start_dist = speed * T * (2.0 * A0 + A1)
        stop_dist = speed * T * (2.0 - 2.0 * A0 - A1)
        cruise_dist = self.args.target_distance_mm - start_dist - stop_dist
        if cruise_dist < 0.0:
            self.get_logger().warn('Target too short for full shaped move; stopping')
            self.state = 'residual'
            self.state_wall = time.monotonic()
            return 0.0, 0.0
        cruise_t = cruise_dist / speed
        if t < T:
            g = A0
        elif t < 2.0 * T:
            g = A0 + A1
        elif t < 2.0 * T + cruise_t:
            g = 1.0
        elif t < 3.0 * T + cruise_t:
            g = 1.0 - A0
        elif t < 4.0 * T + cruise_t:
            g = 1.0 - A0 - A1
        else:
            self.state = 'residual'
            self.state_wall = time.monotonic()
            self.get_logger().info('Fully shaped target command complete; collecting residual')
            return 0.0, 0.0
        return vx * g, vy * g

    def _phase_travel_mm(self):
        start = self.shaped_start_q_mm if self.state in ('shape_move', 'residual', 'done') else self.start_cart_q_mm
        if start is None or self.latest_cart_q_mm is None:
            return 0.0
        direction = 1.0 if (self.args.axis == 'x' and self.args.move_vx_mm_s >= 0.0) or (self.args.axis == 'y' and self.args.move_vy_mm_s >= 0.0) else -1.0
        return direction * (self.latest_cart_q_mm - start)

    def _stream(self, vx, vy):
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = TrajCmd.STREAM
        msg.vx_mm_s = float(vx)
        msg.vy_mm_s = float(vy)
        self.traj_pub.publish(msg)

    def _log(self, vx, vy):
        if self.csv_writer is None:
            return
        est = self.identifier.latest_valid
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds * 1.0e-9,
            self.state,
            time.monotonic() - self.state_wall,
            vx,
            vy,
            '' if self.latest_cart_q_mm is None else self.latest_cart_q_mm,
            '' if self.latest_swing_mm is None else self.latest_swing_mm,
            self._phase_travel_mm(),
            '' if est is None else est.cond_b,
            '' if est is None else est.T,
            '' if est is None else est.zeta,
        ])
        self._flush_count += 1
        if self._flush_count % 25 == 0 and self.csv_file is not None:
            self.csv_file.flush()

    def _print_report(self):
        samples = [x for _, x in self.residual_samples]
        if samples:
            p2p = max(samples) - min(samples)
            max_abs = max(abs(x) for x in samples)
            rms = math.sqrt(sum(x * x for x in samples) / len(samples))
            residual = f'residual p2p={p2p:.2f}mm max_abs={max_abs:.2f}mm rms={rms:.2f}mm samples={len(samples)}'
        else:
            residual = 'residual: no samples'
        est = self.accepted
        est_text = 'fallback used' if est is None else f'T={est.T:.4f}s zeta={est.zeta:.4f} condB={est.cond_b:.3e}'
        self.get_logger().info(
            f'Two-phase adaptive move complete | shaped travel={self._phase_travel_mm():.1f}mm '
            f'target={self.args.target_distance_mm:.1f}mm | {residual} | {est_text}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis', default='x', choices=['x', 'y'])
    p.add_argument('--payload-topic', default='/payload/pose_e_rel')
    p.add_argument('--id-vx-mm-s', type=float, default=200.0)
    p.add_argument('--id-vy-mm-s', type=float, default=0.0)
    p.add_argument('--move-vx-mm-s', type=float, default=200.0)
    p.add_argument('--move-vy-mm-s', type=float, default=0.0)
    p.add_argument('--id-duration', type=float, default=3.0)
    p.add_argument('--min-id-duration', type=float, default=0.8)
    p.add_argument('--settle-after-id', type=float, default=1.0)
    p.add_argument('--target-distance-mm', type=float, default=600.0)
    p.add_argument('--max-travel-mm', type=float, default=900.0)
    p.add_argument('--stream-rate-hz', type=float, default=100.0)
    p.add_argument('--payload-fresh-timeout', type=float, default=0.25)
    p.add_argument('--a0', type=float, default=0.25)
    p.add_argument('--cond-threshold', type=float, default=1.0e8)
    p.add_argument('--accept-cond', type=float, default=1000.0)
    p.add_argument('--omega-min', type=float, default=0.5)
    p.add_argument('--omega-max', type=float, default=20.0)
    p.add_argument('--id-zeta-min', type=float, default=0.0)
    p.add_argument('--id-zeta-max', type=float, default=0.99)
    p.add_argument('--zeta-min', type=float, default=0.0)
    p.add_argument('--zeta-max', type=float, default=0.05)
    p.add_argument('--zv-t-min', type=float, default=0.2)
    p.add_argument('--zv-t-max', type=float, default=1.25)
    p.add_argument('--fallback-t', type=float, default=1.04)
    p.add_argument('--residual-window', type=float, default=2.0)
    p.add_argument('--log-csv', default='')
    return p.parse_args()


def check_args(args):
    if args.axis == 'x' and (abs(args.id_vx_mm_s) < 1e-9 or abs(args.move_vx_mm_s) < 1e-9):
        print('Refusing: x axis needs nonzero x velocities', file=sys.stderr)
        return False
    if args.axis == 'y' and (abs(args.id_vy_mm_s) < 1e-9 or abs(args.move_vy_mm_s) < 1e-9):
        print('Refusing: y axis needs nonzero y velocities', file=sys.stderr)
        return False
    T_guard = args.zv_t_max
    id_travel = math.hypot(args.id_vx_mm_s, args.id_vy_mm_s) * (args.id_duration + T_guard)
    total = id_travel + args.target_distance_mm
    if total > args.max_travel_mm:
        print(f'Refusing: guarded total travel {total:.1f}mm > {args.max_travel_mm:.1f}mm', file=sys.stderr)
        return False
    return True


def main():
    args = parse_args()
    if not check_args(args):
        return 1
    rclpy.init()
    node = None
    try:
        node = TwoPhasePlayer(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[adaptive_two_phase_player] {exc}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
