"""ROS 2 node that shapes live human stick input into gantry STREAM commands."""

from __future__ import annotations

import time
from typing import Sequence

import rclpy
from gantry_control.msg import GantryState, TrajCmd
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .shaper import CausalInputShaper, ImpulseShaper, RateLimiter2D, apply_deadzone


class HmisNode(Node):
    """Map live joystick intent through a causal ZV/ZVD anti-sway filter."""

    def __init__(self) -> None:
        super().__init__('hmis')
        self._declare_parameters()

        shaper_type = str(self.get_parameter('shaper_type').value).lower()
        frequency_hz = float(self.get_parameter('natural_frequency_hz').value)
        damping_ratio = float(self.get_parameter('damping_ratio').value)
        if shaper_type == 'zv':
            impulses = ImpulseShaper.zv(frequency_hz, damping_ratio)
        elif shaper_type == 'zvd':
            impulses = ImpulseShaper.zvd(frequency_hz, damping_ratio)
        else:
            raise ValueError("shaper_type must be 'zv' or 'zvd'")

        self._shaper = CausalInputShaper(impulses)
        self._limiter = RateLimiter2D(
            max_speed=float(self.get_parameter('max_speed_mm_s').value),
            max_acceleration=float(self.get_parameter('max_acceleration_mm_s2').value),
        )
        self._axis_x = int(self.get_parameter('axis_x').value)
        self._axis_y = int(self.get_parameter('axis_y').value)
        self._axis_sign_x = float(self.get_parameter('axis_sign_x').value)
        self._axis_sign_y = float(self.get_parameter('axis_sign_y').value)
        self._deadzone = float(self.get_parameter('deadzone').value)
        self._deadman_button = int(self.get_parameter('deadman_button').value)
        self._estop_button = int(self.get_parameter('estop_button').value)
        self._max_speed = float(self.get_parameter('max_speed_mm_s').value)
        self._joy_timeout_s = float(self.get_parameter('joy_timeout_s').value)
        self._state_timeout_s = float(self.get_parameter('state_timeout_s').value)

        self._latest_joy: Joy | None = None
        self._latest_joy_time: float | None = None
        self._latest_state: GantryState | None = None
        self._latest_state_time: float | None = None
        self._last_stop_reason: str | None = None
        self._estop_button_was_pressed = False

        self._traj_pub = self.create_publisher(TrajCmd, '/traj_cmd', 10)
        self._human_pub = self.create_publisher(Twist, '/hmis/human_cmd', 10)
        self._shaped_pub = self.create_publisher(Twist, '/hmis/shaped_cmd', 10)
        self._status_pub = self.create_publisher(String, '/hmis/status', 10)
        self._estop_client = self.create_client(Trigger, '/gantry/estop')
        self.create_subscription(Joy, '/joy', self._joy_callback, 10)
        self.create_subscription(GantryState, '/gantry/state', self._state_callback, 10)

        rate_hz = float(self.get_parameter('publish_rate_hz').value)
        if rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be positive')
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

        impulse_text = ', '.join(
            f'({delay:.3f}s, {amplitude:.4f})'
            for delay, amplitude in zip(impulses.delays_s, impulses.amplitudes)
        )
        self.get_logger().info(
            f'HMIS ready: {shaper_type.upper()} impulses [{impulse_text}], '
            f'horizon={impulses.horizon_s:.3f}s, max_speed={self._max_speed:.1f}mm/s')
        self.get_logger().info(
            'HMIS never changes mode or enables motors; set TRAJ, home, and enable externally')

    def _declare_parameters(self) -> None:
        parameters = {
            'shaper_type': 'zvd',
            'natural_frequency_hz': 0.5,
            'damping_ratio': 0.0,
            'publish_rate_hz': 100.0,
            'max_speed_mm_s': 60.0,
            'max_acceleration_mm_s2': 150.0,
            'axis_x': 0,
            'axis_y': 1,
            'axis_sign_x': -1.0,
            'axis_sign_y': 1.0,
            'deadzone': 0.08,
            'deadman_button': 4,
            'estop_button': 5,
            'joy_timeout_s': 0.25,
            'state_timeout_s': 0.25,
        }
        for name, default in parameters.items():
            self.declare_parameter(name, default)

    def _joy_callback(self, msg: Joy) -> None:
        self._latest_joy = msg
        self._latest_joy_time = time.monotonic()

    def _state_callback(self, msg: GantryState) -> None:
        self._latest_state = msg
        self._latest_state_time = time.monotonic()

    def _tick(self) -> None:
        now = time.monotonic()
        reason = self._preflight_stop_reason(now)
        if reason is not None:
            if reason == 'operator E-stop':
                self._request_estop_once()
            else:
                self._estop_button_was_pressed = False
            self._hard_stop(now, reason)
            return

        self._estop_button_was_pressed = False
        human = self._map_stick(self._latest_joy.axes)
        limited = self._limiter.limit(now, human)
        shaped = self._shaper.shape(now, limited)
        self._publish_velocity(self._human_pub, limited)
        self._publish_velocity(self._shaped_pub, shaped)
        self._publish_stream(*shaped)
        self._set_status('ACTIVE')

    def _preflight_stop_reason(self, now: float) -> str | None:
        if self._latest_joy is None or self._latest_joy_time is None:
            return 'waiting for joystick'
        if now - self._latest_joy_time > self._joy_timeout_s:
            return 'joystick timeout'
        if self._button(self._latest_joy.buttons, self._estop_button):
            return 'operator E-stop'
        if not self._button(self._latest_joy.buttons, self._deadman_button):
            return 'deadman released'
        if self._latest_state is None or self._latest_state_time is None:
            return 'waiting for gantry state'
        if now - self._latest_state_time > self._state_timeout_s:
            return 'gantry state timeout'
        if self._latest_state.estop:
            return 'gantry E-stop active'
        if not self._latest_state.homed:
            return 'gantry not homed'
        if not self._latest_state.enabled:
            return 'motors not enabled'
        if self._latest_state.mode != 'TRAJ':
            return f'gantry mode is {self._latest_state.mode}, expected TRAJ'
        return None

    def _map_stick(self, axes: Sequence[float]) -> tuple[float, float]:
        required = max(self._axis_x, self._axis_y)
        if required < 0 or len(axes) <= required:
            return 0.0, 0.0
        x = apply_deadzone(self._axis_sign_x * axes[self._axis_x], self._deadzone)
        y = apply_deadzone(self._axis_sign_y * axes[self._axis_y], self._deadzone)
        return x * self._max_speed, y * self._max_speed

    @staticmethod
    def _button(buttons: Sequence[int], index: int) -> bool:
        return 0 <= index < len(buttons) and bool(buttons[index])

    def _hard_stop(self, now: float, reason: str) -> None:
        self._shaper.reset()
        self._limiter.reset(now)
        self._publish_velocity(self._human_pub, (0.0, 0.0))
        self._publish_velocity(self._shaped_pub, (0.0, 0.0))
        self._publish_stream(0.0, 0.0)
        self._set_status(f'STOPPED: {reason}')

    def _request_estop_once(self) -> None:
        if self._estop_button_was_pressed:
            return
        self._estop_button_was_pressed = True
        if not self._estop_client.service_is_ready():
            self.get_logger().error('Operator E-stop pressed but /gantry/estop is unavailable')
            # Retry while the button remains held; STREAM zero is still sent now.
            self._estop_button_was_pressed = False
            return
        self._estop_client.call_async(Trigger.Request())
        self.get_logger().error('Operator E-stop requested')

    def _publish_stream(self, vx_mm_s: float, vy_mm_s: float) -> None:
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = TrajCmd.STREAM
        msg.vx_mm_s = float(vx_mm_s)
        msg.vy_mm_s = float(vy_mm_s)
        self._traj_pub.publish(msg)

    @staticmethod
    def _publish_velocity(publisher, velocity: Sequence[float]) -> None:
        msg = Twist()
        # Keep standard Twist diagnostics in SI; TrajCmd conversion remains internal.
        msg.linear.x = float(velocity[0]) / 1000.0
        msg.linear.y = float(velocity[1]) / 1000.0
        publisher.publish(msg)

    def _set_status(self, status: str) -> None:
        if status == self._last_stop_reason:
            return
        self._last_stop_reason = status
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)
        if status == 'ACTIVE':
            self.get_logger().info(status)
        else:
            self.get_logger().warn(status)

    def destroy_node(self) -> None:
        try:
            for _ in range(3):
                self._publish_stream(0.0, 0.0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HmisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            # A launch supervisor can deliver a second SIGINT during teardown.
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
