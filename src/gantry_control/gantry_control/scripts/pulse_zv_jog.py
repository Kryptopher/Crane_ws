#!/usr/bin/env python3
"""Joystick-selectable direct pulse or nonrobust two-impulse ZV jog.

Button 10 toggles PULSE and ZV while idle.  Hold LB and move the left stick to
jog.  PULSE follows the stick directly and stops immediately on release.  ZV
locks the requested direction and convolves the operator's finite jog command
with a two-impulse rope-mode ZV filter.  It has no LQG, optimal stop, feedback
regulation, or robustness derivative constraint.

RB requests the gantry E-stop.  This node owns /traj_cmd while running, so it
must not run alongside another TRAJ publisher.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from gantry_control.msg import GantryState, TrajCmd
from gantry_control.srv import SetMode
from zv_jog_shaper import make_nonrobust_zv_jog_shaper


class PulseZvJog(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('pulse_zv_jog')
        self.args = args
        self.shaper = make_nonrobust_zv_jog_shaper(
            rope_length_m=args.rope_length_m,
            damping_ratio=args.zv_zeta,
            impulse_spacing_s=args.zv_t_s,
        )
        self.phase = 'arming'
        self.selected_mode = args.start_mode
        self.done = False
        self.latest_gantry: GantryState | None = None
        self.gantry_wall: float | None = None
        self.payload_wall: float | None = None
        self.angle_deg = {'x': None, 'y': None}
        self.joy_wall: float | None = None
        self.joy_direction = {'x': 0.0, 'y': 0.0}
        self.joy_magnitude = 0.0
        self.last_mode_button = False
        self.last_stick_active = False
        self.zv_press_wall: float | None = None
        self.zv_release_wall: float | None = None
        self.armed_wall: float | None = None
        self.last_status_wall = -math.inf
        self._last_csv_flush_wall = time.monotonic()

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.traj_pub = self.create_publisher(TrajCmd, '/traj_cmd', qos)
        self.create_subscription(GantryState, '/gantry/state', self._gantry_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            args.payload_topic,
            self._payload_cb,
            qos_profile_sensor_data,
        )
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
                'wall_time_s', 'selected_mode', 'phase', 'zv_gain',
                'cmd_vx_mm_s', 'cmd_vy_mm_s',
                'cart_x_mm', 'cart_y_mm', 'cart_vx_mm_s', 'cart_vy_mm_s',
                'pitch_deg', 'roll_deg',
            ])

        self.arm_timer = self.create_timer(0.2, self._arm_when_ready)
        self.command_timer = self.create_timer(1.0 / args.rate_hz, self._tick)
        self.get_logger().info(
            f'Pulse/ZV jog ready: start={self.selected_mode.upper()} '
            f'L={args.rope_length_m:.3f}m '
            f'ZV T={self.shaper.impulse_spacing_s:.4f}s '
            f'A=[{self.shaper.first_weight:.4f},'
            f'{self.shaper.second_weight:.4f}] '
            f'speed={args.jog_speed_mm_s:.1f}mm/s. Button 10 toggles '
            'PULSE/ZV while idle; hold LB + left stick to move.'
        )

    @property
    def active_axes(self) -> tuple[str, ...]:
        return ('x', 'y') if self.args.axis == 'both' else (self.args.axis,)

    def _gantry_cb(self, msg: GantryState) -> None:
        self.latest_gantry = msg
        self.gantry_wall = time.monotonic()

    def _payload_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 3:
            return
        pitch = float(msg.data[1])
        roll = float(msg.data[2])
        if not math.isfinite(pitch) or not math.isfinite(roll):
            return
        self.angle_deg['x'] = pitch
        self.angle_deg['y'] = roll
        self.payload_wall = time.monotonic()

    def _ready_to_arm(self) -> bool:
        if self.latest_gantry is None or self.gantry_wall is None:
            return False
        if time.monotonic() - self.gantry_wall > self.args.state_fresh_timeout_s:
            return False
        return bool(
            not self.latest_gantry.estop
            and self.latest_gantry.homed
            and self.payload_wall is not None
            and time.monotonic() - self.payload_wall
            <= self.args.payload_fresh_timeout_s
        )

    def _arm_when_ready(self) -> None:
        if self.done or self.phase != 'arming' or not self._ready_to_arm():
            return
        if not self.mode_cli.wait_for_service(timeout_sec=0.1):
            return
        request = SetMode.Request()
        request.mode = 'TRAJ'
        self.phase = 'arm_pending'
        future = self.mode_cli.call_async(request)
        future.add_done_callback(self._on_set_traj)

    def _on_set_traj(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._stop(f'could not enter TRAJ: {exc}')
            return
        if result is None or not result.success:
            self._stop('gantry rejected TRAJ mode')
            return
        if self.latest_gantry is not None and self.latest_gantry.enabled:
            self._finish_arming()
            return
        if not self.enable_cli.wait_for_service(timeout_sec=1.0):
            self._stop('/gantry/enable unavailable')
            return
        future = self.enable_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_enable)

    def _on_enable(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._stop(f'enable failed: {exc}')
            return
        if result is None or not result.success:
            self._stop('could not enable TRAJ stream')
            return
        self._finish_arming()

    def _finish_arming(self) -> None:
        self.phase = 'idle'
        self.armed_wall = time.monotonic()
        self.get_logger().info(
            f'TRAJ armed. Selected jog mode is {self.selected_mode.upper()}.'
        )

    def _joy_cb(self, msg: Joy) -> None:
        now = time.monotonic()
        self.joy_wall = now
        if self.done or self.phase in ('arming', 'arm_pending'):
            return
        if len(msg.axes) < 2 or len(msg.buttons) < 6:
            self._stop('joystick message is missing required axes/buttons')
            return
        if msg.buttons[5]:
            self._request_estop()
            return

        raw_x = -float(msg.axes[0])
        raw_y = float(msg.axes[1])
        if self.args.axis == 'x':
            raw_y = 0.0
        elif self.args.axis == 'y':
            raw_x = 0.0
        magnitude = math.hypot(raw_x, raw_y)
        lb_held = bool(msg.buttons[4])
        active = lb_held and magnitude > self.args.deadzone

        mode_button = bool(msg.buttons[9]) if len(msg.buttons) > 9 else False
        if mode_button and not self.last_mode_button:
            if self.phase == 'idle' and not active:
                self.selected_mode = (
                    'pulse' if self.selected_mode == 'zv' else 'zv'
                )
                self.get_logger().info(
                    f'Jog mode -> {self.selected_mode.upper()} (Button 10)'
                )
            else:
                self.get_logger().warn(
                    'Release the current jog before changing PULSE/ZV mode'
                )
        self.last_mode_button = mode_button

        if active:
            scale = (magnitude - self.args.deadzone) / max(
                1.0e-9, 1.0 - self.args.deadzone
            )
            scale = max(0.0, min(1.0, scale))
            direction = {
                'x': raw_x / magnitude,
                'y': raw_y / magnitude,
            }
        else:
            scale = 0.0
            direction = {'x': 0.0, 'y': 0.0}

        if self.selected_mode == 'pulse':
            self._handle_pulse_input(active, direction, scale)
        else:
            self._handle_zv_input(active, direction, now)
        self.last_stick_active = active

    def _handle_pulse_input(
        self,
        active: bool,
        direction: dict[str, float],
        magnitude: float,
    ) -> None:
        if active and self.phase in ('idle', 'pulse'):
            self.joy_direction = direction
            self.joy_magnitude = magnitude
            if self.phase == 'idle':
                self.get_logger().info('PULSE jog started')
            self.phase = 'pulse'
        elif not active and self.phase == 'pulse':
            self._publish(0.0, 0.0, 0.0)
            self.phase = 'idle'
            self.joy_magnitude = 0.0
            self.get_logger().info('PULSE jog stopped immediately')

    def _zv_start_is_safe(self) -> tuple[bool, str]:
        if self.latest_gantry is None:
            return False, 'gantry state is unavailable'
        cart_speed = 1000.0 * math.hypot(
            float(self.latest_gantry.vx), float(self.latest_gantry.vy)
        )
        if cart_speed > self.args.start_cart_speed_limit_mm_s:
            return False, (
                f'cart speed {cart_speed:.2f}mm/s exceeds '
                f'{self.args.start_cart_speed_limit_mm_s:.2f}mm/s'
            )
        if (
            self.payload_wall is None
            or time.monotonic() - self.payload_wall
            > self.args.payload_fresh_timeout_s
        ):
            return False, 'payload angle is stale'
        for axis in self.active_axes:
            angle = self.angle_deg[axis]
            if angle is None or abs(angle) > self.args.start_angle_limit_deg:
                return False, (
                    f'{axis}-axis angle is unavailable or exceeds '
                    f'{self.args.start_angle_limit_deg:.3f}deg'
                )
        return True, ''

    def _handle_zv_input(
        self,
        active: bool,
        direction: dict[str, float],
        now: float,
    ) -> None:
        if active and self.phase == 'idle' and not self.last_stick_active:
            safe, reason = self._zv_start_is_safe()
            if not safe:
                self.get_logger().warn(
                    f'ZV jog start refused: {reason}; release and try again'
                )
                return
            self.joy_direction = direction
            self.joy_magnitude = 1.0
            self.zv_press_wall = now
            self.zv_release_wall = None
            self.phase = 'zv'
            self.get_logger().info(
                'ZV jog started: first filtered impulse; direction and speed '
                'are locked until the filtered stop completes'
            )
        elif not active and self.phase == 'zv' and self.zv_release_wall is None:
            self.zv_release_wall = now
            held_s = now - (self.zv_press_wall or now)
            if held_s < self.shaper.impulse_spacing_s:
                self.get_logger().warn(
                    'Short ZV jog released before T; the exact filtered '
                    'delayed half-pulse will still occur'
                )
            self.get_logger().info(
                f'ZV raw jog released after {held_s:.3f}s; playing filtered tail'
            )

    def _tick(self) -> None:
        if self.done or self.phase in ('arming', 'arm_pending'):
            return
        now = time.monotonic()
        if not self._runtime_state_is_safe(now):
            return
        if self.phase in ('pulse', 'zv') and (
            self.joy_wall is None
            or now - self.joy_wall > self.args.joy_fresh_timeout_s
        ):
            self._stop('joystick data became stale during motion')
            return

        if self.phase == 'pulse':
            speed = self.args.jog_speed_mm_s * self.joy_magnitude
            self._publish(
                self.joy_direction['x'] * speed,
                self.joy_direction['y'] * speed,
                1.0,
            )
        elif self.phase == 'zv':
            self._publish_zv(now)
        else:
            self._publish(0.0, 0.0, 0.0)

    def _runtime_state_is_safe(self, now: float) -> bool:
        if (
            self.latest_gantry is None
            or self.gantry_wall is None
            or now - self.gantry_wall > self.args.state_fresh_timeout_s
        ):
            self._stop('gantry state became stale')
            return False
        if self.latest_gantry.estop:
            self._stop('gantry E-stop is active')
            return False
        if not self.latest_gantry.homed:
            self._stop('gantry is no longer homed')
            return False
        if self.armed_wall is not None and now - self.armed_wall >= 0.5:
            if not self.latest_gantry.enabled:
                self._stop('motors were disabled')
                return False
            if self.latest_gantry.mode != 'TRAJ':
                self._stop(f'gantry left TRAJ mode ({self.latest_gantry.mode})')
                return False
        return True

    def _publish_zv(self, now: float) -> None:
        if self.zv_press_wall is None:
            self._stop('ZV timing state is unavailable')
            return
        elapsed = now - self.zv_press_wall
        release_elapsed = (
            None
            if self.zv_release_wall is None
            else self.zv_release_wall - self.zv_press_wall
        )
        gain = self.shaper.gain_at(elapsed, release_elapsed)
        speed = gain * self.args.jog_speed_mm_s
        self._publish(
            self.joy_direction['x'] * speed,
            self.joy_direction['y'] * speed,
            gain,
        )
        if self.shaper.is_complete(elapsed, release_elapsed):
            self._publish(0.0, 0.0, 0.0)
            self.phase = 'idle'
            self.zv_press_wall = None
            self.zv_release_wall = None
            self.joy_magnitude = 0.0
            self.get_logger().info(
                'ZV filtered stop complete; command is zero and jog remains armed'
            )

    def _workspace_allows(self, vx: float, vy: float) -> bool:
        if self.latest_gantry is None:
            return False
        x_mm = 1000.0 * float(self.latest_gantry.x)
        y_mm = 1000.0 * float(self.latest_gantry.y)
        low = self.args.workspace_min_mm + self.args.workspace_margin_mm
        high = self.args.workspace_max_mm - self.args.workspace_margin_mm
        return not (
            (vx < 0.0 and x_mm <= low)
            or (vx > 0.0 and x_mm >= high)
            or (vy < 0.0 and y_mm <= low)
            or (vy > 0.0 and y_mm >= high)
        )

    def _publish(self, vx: float, vy: float, gain: float) -> None:
        if (vx != 0.0 or vy != 0.0) and not self._workspace_allows(vx, vy):
            self._stop('workspace guard blocked the commanded direction')
            return
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = TrajCmd.STREAM
        msg.vx_mm_s = float(vx)
        msg.vy_mm_s = float(vy)
        if self.latest_gantry is not None:
            msg.x = float(self.latest_gantry.x)
            msg.y = float(self.latest_gantry.y)
        self.traj_pub.publish(msg)
        self._log(vx, vy, gain)

    def _log(self, vx: float, vy: float, gain: float) -> None:
        if self.csv_writer is None:
            return
        state = self.latest_gantry
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds * 1.0e-9,
            self.selected_mode,
            self.phase,
            gain,
            vx,
            vy,
            '' if state is None else 1000.0 * float(state.x),
            '' if state is None else 1000.0 * float(state.y),
            '' if state is None else 1000.0 * float(state.vx),
            '' if state is None else 1000.0 * float(state.vy),
            '' if self.angle_deg['x'] is None else self.angle_deg['x'],
            '' if self.angle_deg['y'] is None else self.angle_deg['y'],
        ])
        now = time.monotonic()
        if self.csv_file is not None and now - self._last_csv_flush_wall >= 0.5:
            self.csv_file.flush()
            self._last_csv_flush_wall = now

    def _request_estop(self) -> None:
        self._publish(0.0, 0.0, 0.0)
        if self.estop_cli.wait_for_service(timeout_sec=0.2):
            self.estop_cli.call_async(Trigger.Request())
        self._stop('RB pressed: E-stop requested')

    def _stop(self, reason: str) -> None:
        if self.done:
            return
        try:
            msg = TrajCmd()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.command = TrajCmd.STREAM
            msg.vx_mm_s = 0.0
            msg.vy_mm_s = 0.0
            self.traj_pub.publish(msg)
        except Exception:
            pass
        if self.csv_file is not None:
            self.csv_file.flush()
        self.get_logger().info(f'Pulse/ZV jog finished: {reason}')
        self.done = True

    def destroy_node(self) -> None:
        if not self.done:
            self._stop('node shutdown')
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis', choices=('x', 'y', 'both'), default='both')
    parser.add_argument(
        '--start-mode',
        choices=('pulse', 'zv', 'hybrid'),
        default='zv',
        help='Initial mode. Legacy "hybrid" is accepted as an alias for "zv".',
    )
    parser.add_argument('--payload-topic', default='/payload/pose_e_rel')
    parser.add_argument('--rope-length-m', type=float, default=0.90)
    parser.add_argument(
        '--zv-t-s', '--zvd-t-s', dest='zv_t_s', type=float, default=0.0,
        help='ZV impulse spacing; <=0 uses pi*sqrt(L/g).',
    )
    parser.add_argument(
        '--zv-zeta', '--zvd-zeta', dest='zv_zeta', type=float, default=0.0,
    )
    parser.add_argument('--jog-speed-mm-s', type=float, default=100.0)
    parser.add_argument('--rate-hz', type=float, default=100.0)
    parser.add_argument('--deadzone', type=float, default=0.08)
    parser.add_argument('--start-angle-limit-deg', type=float, default=0.50)
    parser.add_argument('--start-cart-speed-limit-mm-s', type=float, default=5.0)
    parser.add_argument('--payload-fresh-timeout-s', '--payload-fresh-timeout',
                        dest='payload_fresh_timeout_s', type=float, default=0.30)
    parser.add_argument('--state-fresh-timeout-s', type=float, default=0.30)
    parser.add_argument('--joy-fresh-timeout-s', type=float, default=0.30)
    parser.add_argument('--workspace-min-mm', type=float, default=0.0)
    parser.add_argument('--workspace-max-mm', type=float, default=1150.0)
    parser.add_argument('--workspace-margin-mm', type=float, default=5.0)
    parser.add_argument('--log-csv', default='')

    # Preserve old launch commands while deliberately ignoring every LQG-only
    # tuning option.  The executable name remains a compatibility alias below.
    legacy_float_options = (
        '--max-regulate-vel-mm-s', '--max-regulate-accel-mm-s2',
        '--final-cart-tolerance-mm', '--done-rms-deg', '--done-max-abs-deg',
        '--done-window-s', '--min-regulate-s', '--quiescent-angle-deg',
        '--quiescent-angle-rate-deg-s', '--quiescent-cart-vel-mm-s',
        '--quiescent-hold-s', '--command-deadband-angle-deg',
        '--min-effective-command-mm-s', '--reversal-command-threshold-mm-s',
        '--hunt-window-s', '--hunt-angle-deg', '--hunt-position-error-mm',
        '--model-zeta', '--actuator-tau-s', '--angle-measurement-std-deg',
        '--q-cart-position', '--q-cart-velocity', '--q-angle', '--q-angle-rate',
        '--r-command', '--print-period',
    )
    for option in legacy_float_options:
        parser.add_argument(option, type=float, help=argparse.SUPPRESS)
    parser.add_argument('--hunt-reversal-limit', type=int, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.start_mode == 'hybrid':
        args.start_mode = 'zv'
    return args


def check_args(args: argparse.Namespace) -> None:
    positive = {
        'rope-length-m': args.rope_length_m,
        'jog-speed-mm-s': args.jog_speed_mm_s,
        'rate-hz': args.rate_hz,
        'start-angle-limit-deg': args.start_angle_limit_deg,
        'start-cart-speed-limit-mm-s': args.start_cart_speed_limit_mm_s,
        'payload-fresh-timeout-s': args.payload_fresh_timeout_s,
        'state-fresh-timeout-s': args.state_fresh_timeout_s,
        'joy-fresh-timeout-s': args.joy_fresh_timeout_s,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'--{name} must be finite and positive')
    if args.jog_speed_mm_s > 1000.0:
        raise ValueError('--jog-speed-mm-s cannot exceed controller limit 1000mm/s')
    if not 0.0 <= args.zv_zeta < 1.0:
        raise ValueError('--zv-zeta must be in [0, 1)')
    if not 0.0 <= args.deadzone < 1.0:
        raise ValueError('--deadzone must be in [0, 1)')
    if args.zv_t_s < 0.0:
        raise ValueError('--zv-t-s must be nonnegative')
    if (
        args.workspace_margin_mm < 0.0
        or args.workspace_max_mm
        <= args.workspace_min_mm + 2.0 * args.workspace_margin_mm
    ):
        raise ValueError('invalid workspace bounds or margin')


def main() -> int:
    args = parse_args()
    try:
        check_args(args)
    except ValueError as exc:
        print(f'[pulse_zv_jog] {exc}', file=sys.stderr)
        return 2
    rclpy.init()
    node = PulseZvJog(args)
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
