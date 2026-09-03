#!/usr/bin/env python3
"""
Initial-condition cart serial node — X-axis release cart.

Wraps the Arduino Nano driving a NEMA17 + TB6600 through a GT2 20T pulley
(arduino_nano_ic_cart_stepper.ino).  Firmware protocol, 115200 baud:

  G<mm>      absolute move        S          stop
  Z          zero origin          V<mm/s>    cruise speed
  E0 / E1    force disable/enable driver

Telemetry in, 50 Hz CSV: millis,step_pos_mm,enc_pos_mm,target_mm,moving

  ros2 run gantry_control ic_cart_node.py
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger

from gantry_control.srv import IcCartMove

try:
    import serial
except ImportError:
    serial = None

STATE_FIELDS = [
    't', 'step_pos_mm', 'enc_pos_mm', 'target_mm', 'moving', 'arduino_ms', 'stale',
]


class IcCartNode(Node):
    def __init__(self) -> None:
        super().__init__('ic_cart_node')

        self.declare_parameter('serial_port', '/dev/ttyCH341USB1')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('cruise_speed_mm_s', 15.0)
        self.declare_parameter('min_position_mm', -1000.0)
        self.declare_parameter('max_position_mm', 1000.0)
        self.declare_parameter('max_velocity_mm_s', 60.0)
        self.declare_parameter('stale_timeout_s', 0.5)

        self._port = str(self.get_parameter('serial_port').value)
        self._baud = int(self.get_parameter('baud').value)
        self._publish_hz = max(5.0, min(
            float(self.get_parameter('publish_rate_hz').value), 100.0))
        self._min_pos_mm = float(self.get_parameter('min_position_mm').value)
        self._max_pos_mm = float(self.get_parameter('max_position_mm').value)
        # Firmware clamps to its own MAX_SPEED_MM_S; keep this in sync with it.
        self._max_vel_mm_s = max(1.0, float(
            self.get_parameter('max_velocity_mm_s').value))
        self._cruise_speed_mm_s = self._clamp_velocity(
            float(self.get_parameter('cruise_speed_mm_s').value))
        self._stale_timeout_s = max(
            0.05, float(self.get_parameter('stale_timeout_s').value))

        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._ser = None
        self._arduino_ms = 0.0
        self._step_pos_mm = 0.0
        self._enc_pos_mm = 0.0
        self._target_mm = 0.0
        self._moving = 0.0
        self._last_rx_mono = 0.0
        self._serial_lines = 0
        self._parse_errors = 0
        self._t0 = time.monotonic()

        self.pub = self.create_publisher(
            Float64MultiArray, '/ic_cart/state', qos_profile_sensor_data)

        self.create_service(
            IcCartMove, '/ic_cart/move_to', self._on_move_to)
        self.create_service(
            Trigger, '/ic_cart/calibrate_origin', self._on_calibrate_origin)
        self.create_service(Trigger, '/ic_cart/stop', self._on_stop)
        self.create_service(Trigger, '/ic_cart/enable', self._on_enable)
        self.create_service(Trigger, '/ic_cart/disable', self._on_disable)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        if serial is None:
            raise RuntimeError('python3-serial/pyserial is not installed')

        self._ser = self._open_serial_with_retry()
        self._send('V%.2f' % self._cruise_speed_mm_s)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._timer = self.create_timer(1.0 / self._publish_hz, self._publish)

        self.get_logger().info(
            f'IC cart ready port={self._port} baud={self._baud} '
            f'@ {self._publish_hz:.1f} Hz '
            f'v={self._cruise_speed_mm_s:.1f} mm/s '
            f'limits=[{self._min_pos_mm:.0f}, {self._max_pos_mm:.0f}] mm')

    def _clamp_velocity(self, value: float) -> float:
        return max(1.0, min(float(value), self._max_vel_mm_s))

    def _open_serial_with_retry(self):
        # No glob fallback here on purpose: the payload encoder Nano is an
        # identical CH340 with the same VID:PID and no by-id symlink, so
        # autodetection could bind this node to the encoder and start writing
        # motion commands at it. Fail loudly on the configured port instead.
        deadline = time.monotonic() + 12.0
        last_exc: Optional[Exception] = None
        while time.monotonic() < deadline and not self._stop.is_set():
            try:
                ser = serial.Serial(self._port, self._baud, timeout=0.1)
            except Exception as exc:
                last_exc = exc
                self.get_logger().warn(
                    f'Waiting for IC cart serial port {self._port}: {exc}')
                time.sleep(0.5)
                continue
            if self._verify_ic_cart(ser):
                return ser
            ser.close()
            raise RuntimeError(
                f'{self._port} is not the IC cart Nano — expected 5-field '
                f'telemetry (millis,step_pos_mm,enc_pos_mm,target_mm,moving). '
                'Check which CH340 enumerated where.')

        if last_exc is not None:
            raise RuntimeError(
                f'could not open IC cart serial port {self._port} '
                f'after 12s: {last_exc}')
        raise RuntimeError(f'could not open IC cart serial port {self._port}')

    def _verify_ic_cart(self, ser) -> bool:
        """Confirm the port is streaming IC-cart telemetry, not encoder data."""
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                raw = ser.readline()
            except Exception:
                return False
            line = raw.decode('ascii', errors='ignore').strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 5:
                try:
                    [float(p) for p in parts]
                except ValueError:
                    continue
                return True
        return False

    def _send(self, command: str) -> bool:
        if self._ser is None:
            return False
        try:
            with self._write_lock:
                self._ser.write((command + '\n').encode('ascii'))
            return True
        except Exception as exc:
            self.get_logger().warn(f'Serial write "{command}" failed: {exc}')
            return False

    def _on_set_parameters(self, params):
        for param in params:
            if param.name == 'cruise_speed_mm_s':
                value = float(param.value)
                if value <= 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason='cruise_speed_mm_s must be > 0',
                    )
                clamped = self._clamp_velocity(value)
                if not self._send('V%.2f' % clamped):
                    return SetParametersResult(
                        successful=False,
                        reason='serial write failed',
                    )
                self._cruise_speed_mm_s = clamped
                self.get_logger().info(
                    f'cruise_speed_mm_s updated to {clamped:.2f} mm/s')
        return SetParametersResult(successful=True)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._ser.readline()
            except Exception as exc:
                self.get_logger().warn(f'Serial read failed: {exc}')
                time.sleep(0.25)
                continue
            if not raw:
                continue
            line = raw.decode('ascii', errors='ignore').strip()
            if not line:
                continue
            try:
                parts = line.split(',')
                if len(parts) != 5:
                    raise ValueError('expected 5 comma-separated fields')
                arduino_ms = float(parts[0])
                step_pos_mm = float(parts[1])
                enc_pos_mm = float(parts[2])
                target_mm = float(parts[3])
                moving = float(parts[4])
            except (ValueError, IndexError):
                with self._lock:
                    self._parse_errors += 1
                continue

            with self._lock:
                self._arduino_ms = arduino_ms
                self._step_pos_mm = step_pos_mm
                self._enc_pos_mm = enc_pos_mm
                self._target_mm = target_mm
                self._moving = moving
                self._last_rx_mono = time.monotonic()
                self._serial_lines += 1

    def _publish(self) -> None:
        now = time.monotonic()
        with self._lock:
            arduino_ms = self._arduino_ms
            step_pos_mm = self._step_pos_mm
            enc_pos_mm = self._enc_pos_mm
            target_mm = self._target_mm
            moving = self._moving
            last_rx = self._last_rx_mono
        stale = last_rx <= 0.0 or (now - last_rx) > self._stale_timeout_s

        msg = Float64MultiArray()
        dim = MultiArrayDimension()
        dim.label = ','.join(STATE_FIELDS)
        dim.size = len(STATE_FIELDS)
        dim.stride = len(STATE_FIELDS)
        msg.layout.dim.append(dim)
        msg.data = [
            now - self._t0,
            step_pos_mm,
            enc_pos_mm,
            target_mm,
            moving,
            arduino_ms,
            1.0 if stale else 0.0,
        ]
        self.pub.publish(msg)

    def _on_move_to(self, request, response):
        position_mm = float(request.position_mm)
        if not (self._min_pos_mm <= position_mm <= self._max_pos_mm):
            response.success = False
            response.message = (
                f'{position_mm:.1f} mm outside travel limits '
                f'[{self._min_pos_mm:.0f}, {self._max_pos_mm:.0f}] mm')
            return response

        velocity_mm_s = float(request.velocity_mm_s)
        if velocity_mm_s > 0.0:
            clamped = self._clamp_velocity(velocity_mm_s)
            if not self._send('V%.2f' % clamped):
                response.success = False
                response.message = 'serial write failed setting velocity'
                return response
            self._cruise_speed_mm_s = clamped

        if not self._send('G%.2f' % position_mm):
            response.success = False
            response.message = 'serial write failed sending move'
            return response

        response.success = True
        response.message = (
            f'Moving to {position_mm:.1f} mm '
            f'@ {self._cruise_speed_mm_s:.1f} mm/s')
        self.get_logger().info(response.message)
        return response

    def _on_calibrate_origin(self, _request, response):
        ok = self._send('Z')
        response.success = ok
        response.message = (
            'IC cart origin set at current position' if ok
            else 'serial write failed')
        if ok:
            self.get_logger().info('/ic_cart/calibrate_origin — origin zeroed')
        return response

    def _on_stop(self, _request, response):
        ok = self._send('S')
        response.success = ok
        response.message = 'IC cart stopped' if ok else 'serial write failed'
        return response

    def _on_enable(self, _request, response):
        ok = self._send('E1')
        response.success = ok
        response.message = (
            'IC cart driver enabled' if ok else 'serial write failed')
        return response

    def _on_disable(self, _request, response):
        ok = self._send('E0')
        response.success = ok
        response.message = (
            'IC cart driver disabled' if ok else 'serial write failed')
        return response

    def destroy_node(self):
        if self._ser is not None:
            self._send('S')
        self._stop.set()
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node: Optional[IcCartNode] = None
    try:
        node = IcCartNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[ic_cart_node] Fatal: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
