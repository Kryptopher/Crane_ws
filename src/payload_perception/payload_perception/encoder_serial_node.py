#!/usr/bin/env python3
"""
encoder_serial_node — Arduino Nano quadrature encoder bridge.

Reads lines from USB serial, either the original encoder-only format:
  millis,pitch_count,roll_count

or the current encoder+gyro(dps)+radio-status format:
  millis,pitch_count,roll_count,gx1_dps,gy1_dps,gz1_dps,gx2_dps,gy2_dps,gz2_dps,packet_age_ms,packet_seen

or the legacy encoder+IMU(accel+gyro raw counts)[+radio-status] format, kept for
backward compatibility (accel/gyro here are raw counts, not dps — imu*_dps
fields are published as NaN for these lines since there's no known conversion
factor from raw counts to dps):
  millis,pitch_count,roll_count,ax1,ay1,az1,gx1,gy1,gz1,ax2,ay2,az2,gx2,gy2,gz2[,packet_age_ms,packet_seen]

Publishes the existing dashboard/logger-compatible topics:
  /payload/pose_e
  /payload/pose_e_rel
  /payload/encoder/diagnostics
"""

from __future__ import annotations

import math
import glob
import os
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger

try:
  import serial
except ImportError:  # pragma: no cover - runtime dependency on Jetson
  serial = None


POSE_E_FIELDS = ('time', 'pitch_deg', 'roll_deg', 'pitch_count', 'roll_count')
POSE_E_REL_FIELDS = (
  'time', 'pitch_deg', 'roll_deg',
  'x_rel_m', 'y_rel_m', 'z_rel_m',
  'vx_rel_m_s', 'vy_rel_m_s', 'vz_rel_m_s',
)
IMU_RAW_FIELDS = (
  'time',
  'arduino_ms',
  'imu1_gx_dps',
  'imu1_gy_dps',
  'imu1_gz_dps',
  'imu2_gx_dps',
  'imu2_gy_dps',
  'imu2_gz_dps',
  'packet_age_ms',
  'packet_seen',
)
DIAG_FIELDS = (
  'time',
  'arduino_ms',
  'pitch_raw',
  'roll_raw',
  'pitch_count',
  'roll_count',
  'imu1_gx_dps',
  'imu1_gy_dps',
  'imu1_gz_dps',
  'imu2_gx_dps',
  'imu2_gy_dps',
  'imu2_gz_dps',
  'packet_age_ms',
  'packet_seen',
  'serial_lines',
  'parse_errors',
  'stale',
)


class EncoderSerialNode(Node):
  def __init__(self):
    super().__init__('encoder_serial_node')

    self.declare_parameter('serial_port', '/dev/ttyACM0')
    self.declare_parameter('baud', 115200)
    self.declare_parameter('publish_rate_hz', 100.0)
    self.declare_parameter('zero_on_start', True)
    self.declare_parameter('ppr', 1000)
    self.declare_parameter('count_mode', 4)
    self.declare_parameter('invert_pitch', False)
    self.declare_parameter('invert_roll', False)
    self.declare_parameter('rope_length_m', 1.20)
    self.declare_parameter('rel_vel_ema_alpha', 0.35)
    self.declare_parameter('rel_vel_min_dt_s', 0.003)
    self.declare_parameter('rel_sign_x', 1.0)
    self.declare_parameter('rel_sign_y', 1.0)
    self.declare_parameter('stale_timeout_s', 0.5)

    self._configured_port = str(self.get_parameter('serial_port').value)
    self._port = self._resolve_serial_port(self._configured_port)
    self._baud = int(self.get_parameter('baud').value)
    publish_hz = float(self.get_parameter('publish_rate_hz').value)
    self._publish_hz = max(5.0, min(publish_hz, 200.0))
    self._zero_on_start = bool(self.get_parameter('zero_on_start').value)
    ppr = int(self.get_parameter('ppr').value)
    count_mode = int(self.get_parameter('count_mode').value)
    self._deg_per_count = 360.0 / max(1, ppr * count_mode)
    self._invert_pitch = bool(self.get_parameter('invert_pitch').value)
    self._invert_roll = bool(self.get_parameter('invert_roll').value)
    self._rope_length_m = max(0.05, float(self.get_parameter('rope_length_m').value))
    self._rel_vel_alpha = max(
      0.0, min(float(self.get_parameter('rel_vel_ema_alpha').value), 1.0))
    self._rel_vel_min_dt_s = max(
      1e-4, float(self.get_parameter('rel_vel_min_dt_s').value))
    self._rel_sign_x = (
      1.0 if float(self.get_parameter('rel_sign_x').value) >= 0.0 else -1.0)
    self._rel_sign_y = (
      1.0 if float(self.get_parameter('rel_sign_y').value) >= 0.0 else -1.0)
    self._stale_timeout_s = max(
      0.05, float(self.get_parameter('stale_timeout_s').value))

    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._ser = None
    self._arduino_ms = 0
    self._pitch_raw = 0
    self._roll_raw = 0
    self._gyro_dps = [float('nan')] * 6
    self._packet_age_ms = float('nan')
    self._packet_seen = float('nan')
    self._pitch_zero: Optional[int] = None
    self._roll_zero: Optional[int] = None
    self._last_rx_mono = 0.0
    self._serial_lines = 0
    self._parse_errors = 0
    self._prev_rel_t: Optional[float] = None
    self._prev_rel_pos: Optional[Tuple[float, float, float]] = None
    self._prev_rel_vel: Optional[Tuple[float, float, float]] = None
    self._t0 = time.monotonic()

    self.pub = self.create_publisher(
      Float64MultiArray, '/payload/pose_e', qos_profile_sensor_data)
    self.pub_rel = self.create_publisher(
      Float64MultiArray, '/payload/pose_e_rel', qos_profile_sensor_data)
    self.pub_imu = self.create_publisher(
      Float64MultiArray, '/payload/imu_raw', qos_profile_sensor_data)
    self.pub_diag = self.create_publisher(
      Float64MultiArray, '/payload/encoder/diagnostics', qos_profile_sensor_data)
    self.create_service(
      Trigger, '/payload/encoder/reset_origin', self._on_reset_origin)
    self.add_on_set_parameters_callback(self._on_set_parameters)

    if serial is None:
      raise RuntimeError('python3-serial/pyserial is not installed')

    self._ser = self._open_serial_with_retry()
    self._reader = threading.Thread(target=self._read_loop, daemon=True)
    self._reader.start()
    self._timer = self.create_timer(1.0 / self._publish_hz, self._publish)

    self.get_logger().info(
      f'Arduino encoder serial ready port={self._port} baud={self._baud} '
      f'@ {self._publish_hz:.1f} Hz deg/count={self._deg_per_count:.5f} '
      f'(L={self._rope_length_m:.2f} m)')

  def _open_serial_with_retry(self):
    deadline = time.monotonic() + 12.0
    last_exc: Exception | None = None
    last_port = self._port
    while time.monotonic() < deadline:
      port = self._resolve_serial_port(self._configured_port)
      last_port = port
      if port != self._configured_port:
        self.get_logger().warn(
          f'Configured serial port {self._configured_port} is unavailable; '
          f'using detected port {port}')
      try:
        ser = serial.Serial(port, self._baud, timeout=0.1)
      except Exception as exc:
        last_exc = exc
        self.get_logger().warn(
          f'Waiting for encoder serial port {self._configured_port} '
          f'(last tried {port}): {exc}')
        time.sleep(0.5)
        continue
      self._port = port
      return ser

    if last_exc is not None:
      raise RuntimeError(
        f'could not open encoder serial port {self._configured_port} '
        f'(last tried {last_port}) after 12s: {last_exc}')
    raise RuntimeError(
      f'could not find encoder serial port {self._configured_port} after 12s')

  def _resolve_serial_port(self, configured_port: str) -> str:
    if os.path.exists(configured_port):
      return configured_port

    patterns = [
      '/dev/ttyCH341USB*',
      '/dev/serial/by-id/*',
      '/dev/serial/by-path/*',
      '/dev/ttyUSB*',
      '/dev/ttyACM*',
    ]
    candidates = []
    seen = set()
    for pattern in patterns:
      for path in sorted(glob.glob(pattern)):
        if path in seen:
          continue
        seen.add(path)
        if os.path.exists(path):
          candidates.append(path)

    if len(candidates) == 1:
      return candidates[0]

    ch341_candidates = [
      path for path in candidates
      if os.path.basename(path).startswith('ttyCH341USB')
    ]
    if len(ch341_candidates) == 1:
      return ch341_candidates[0]

    if candidates:
      raise RuntimeError(
        f'Configured serial port {configured_port} is unavailable and multiple '
        f'candidate serial ports were found: {", ".join(candidates)}. '
        'Set encoder_serial_node.serial_port explicitly.')

    return configured_port

  def _on_set_parameters(self, params):
    for param in params:
      if param.name == 'rope_length_m':
        value = float(param.value)
        if value < 0.05:
          return SetParametersResult(
            successful=False,
            reason='rope_length_m must be >= 0.05 m',
          )
        self._rope_length_m = value
        self.get_logger().info(f'rope_length_m updated to {value:.3f} m')
      elif param.name == 'rel_sign_x':
        self._rel_sign_x = 1.0 if float(param.value) >= 0.0 else -1.0
      elif param.name == 'rel_sign_y':
        self._rel_sign_y = 1.0 if float(param.value) >= 0.0 else -1.0
    return SetParametersResult(successful=True)

  def _read_loop(self):
    while not self._stop.is_set():
      try:
        raw = self._ser.readline()
      except Exception as exc:
        self.get_logger().warn(f'Serial read failed: {exc}')
        time.sleep(0.25)
        continue
      if not raw:
        continue
      try:
        line = raw.decode('ascii', errors='ignore').strip()
        if not line or line.lower().startswith('time_ms,'):
          continue
        parts = line.split(',')
        if len(parts) not in (3, 11, 15, 17):
          raise ValueError('expected 3, 11, 15, or 17 comma-separated fields')
        arduino_ms = float(parts[0])
        pitch_raw = int(parts[1])
        roll_raw = int(parts[2])
        if len(parts) == 11:
          gyro_dps = [float(value) for value in parts[3:9]]
          packet_age_ms = float(parts[9])
          packet_seen = float(parts[10])
        else:
          # Legacy accel+gyro raw-count format (or encoder-only): no dps
          # reading available.
          gyro_dps = [float('nan')] * 6
          packet_age_ms = float(parts[15]) if len(parts) >= 17 else float('nan')
          packet_seen = float(parts[16]) if len(parts) >= 17 else float('nan')
      except Exception:
        with self._lock:
          self._parse_errors += 1
        continue

      with self._lock:
        self._arduino_ms = arduino_ms
        self._pitch_raw = pitch_raw
        self._roll_raw = roll_raw
        self._gyro_dps = gyro_dps
        self._packet_age_ms = packet_age_ms
        self._packet_seen = packet_seen
        if self._zero_on_start and self._pitch_zero is None:
          self._pitch_zero = pitch_raw
          self._roll_zero = roll_raw
        self._last_rx_mono = time.monotonic()
        self._serial_lines += 1

  def _relative_counts_locked(self) -> Tuple[int, int]:
    pitch_zero = self._pitch_zero if self._pitch_zero is not None else 0
    roll_zero = self._roll_zero if self._roll_zero is not None else 0
    pitch = int(self._pitch_raw - pitch_zero)
    roll = int(self._roll_raw - roll_zero)
    if self._invert_pitch:
      pitch *= -1
    if self._invert_roll:
      roll *= -1
    return pitch, roll

  def _counts_to_degrees(self, counts: int) -> float:
    return float(counts) * self._deg_per_count

  def _payload_relative_from_angles(
    self, pitch_deg: float, roll_deg: float
  ) -> Tuple[float, float, float]:
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    L = self._rope_length_m
    x_rel = self._rel_sign_x * L * math.sin(pitch)
    y_rel = self._rel_sign_y * L * math.sin(roll)
    z_rel = -L * math.cos(pitch) * math.cos(roll)
    return x_rel, y_rel, z_rel

  def _payload_relative_velocity(
    self, t: float, pos: Tuple[float, float, float]
  ) -> Tuple[float, float, float]:
    if self._prev_rel_t is None or self._prev_rel_pos is None:
      self._prev_rel_t = t
      self._prev_rel_pos = pos
      return float('nan'), float('nan'), float('nan')
    dt = t - self._prev_rel_t
    if dt < self._rel_vel_min_dt_s:
      return self._prev_rel_vel or (float('nan'), float('nan'), float('nan'))
    vx = (pos[0] - self._prev_rel_pos[0]) / dt
    vy = (pos[1] - self._prev_rel_pos[1]) / dt
    vz = (pos[2] - self._prev_rel_pos[2]) / dt
    if self._prev_rel_vel is not None and self._rel_vel_alpha > 0.0:
      a = self._rel_vel_alpha
      vx = a * vx + (1.0 - a) * self._prev_rel_vel[0]
      vy = a * vy + (1.0 - a) * self._prev_rel_vel[1]
      vz = a * vz + (1.0 - a) * self._prev_rel_vel[2]
    self._prev_rel_t = t
    self._prev_rel_pos = pos
    self._prev_rel_vel = (vx, vy, vz)
    return vx, vy, vz

  def _build_rel_msg(
    self, t: float, pitch_deg: float, roll_deg: float
  ) -> Float64MultiArray:
    x_rel, y_rel, z_rel = self._payload_relative_from_angles(pitch_deg, roll_deg)
    vx_rel, vy_rel, vz_rel = self._payload_relative_velocity(
      t, (x_rel, y_rel, z_rel))
    msg = Float64MultiArray()
    dim = MultiArrayDimension()
    dim.label = ','.join(POSE_E_REL_FIELDS)
    dim.size = len(POSE_E_REL_FIELDS)
    dim.stride = len(POSE_E_REL_FIELDS)
    msg.layout.dim.append(dim)
    msg.data = [
      float(t), float(pitch_deg), float(roll_deg),
      float(x_rel), float(y_rel), float(z_rel),
      float(vx_rel), float(vy_rel), float(vz_rel),
    ]
    return msg

  def _build_imu_msg(
    self,
    t: float,
    arduino_ms: float,
    gyro_dps: list[float],
    packet_age_ms: float,
    packet_seen: float,
  ) -> Float64MultiArray:
    msg = Float64MultiArray()
    dim = MultiArrayDimension()
    dim.label = ','.join(IMU_RAW_FIELDS)
    dim.size = len(IMU_RAW_FIELDS)
    dim.stride = len(IMU_RAW_FIELDS)
    msg.layout.dim.append(dim)
    values = list(gyro_dps[:6])
    if len(values) < 6:
      values.extend([float('nan')] * (6 - len(values)))
    msg.data = [
      float(t),
      float(arduino_ms),
      *[float(value) for value in values],
      float(packet_age_ms),
      float(packet_seen),
    ]
    return msg

  def _publish(self):
    now = time.monotonic()
    with self._lock:
      arduino_ms = self._arduino_ms
      pitch_raw = self._pitch_raw
      roll_raw = self._roll_raw
      gyro_dps = list(self._gyro_dps)
      packet_age_ms = self._packet_age_ms
      packet_seen = self._packet_seen
      pitch, roll = self._relative_counts_locked()
      serial_lines = self._serial_lines
      parse_errors = self._parse_errors
      stale = self._last_rx_mono <= 0.0 or now - self._last_rx_mono > self._stale_timeout_s

    t = now - self._t0
    pitch_deg = self._counts_to_degrees(pitch)
    roll_deg = self._counts_to_degrees(roll)

    msg = Float64MultiArray()
    dim = MultiArrayDimension()
    dim.label = ','.join(POSE_E_FIELDS)
    dim.size = len(POSE_E_FIELDS)
    dim.stride = len(POSE_E_FIELDS)
    msg.layout.dim.append(dim)
    msg.data = [
      float(t), pitch_deg, roll_deg, float(pitch), float(roll),
    ]
    self.pub.publish(msg)
    self.pub_rel.publish(self._build_rel_msg(t, pitch_deg, roll_deg))
    self.pub_imu.publish(
      self._build_imu_msg(
        t, arduino_ms, gyro_dps, packet_age_ms, packet_seen))

    diag = Float64MultiArray()
    dim = MultiArrayDimension()
    dim.label = ','.join(DIAG_FIELDS)
    dim.size = len(DIAG_FIELDS)
    dim.stride = len(DIAG_FIELDS)
    diag.layout.dim.append(dim)
    diag.data = [
      float(t),
      float(arduino_ms),
      float(pitch_raw),
      float(roll_raw),
      float(pitch),
      float(roll),
      *[float(value) for value in gyro_dps],
      float(packet_age_ms),
      float(packet_seen),
      float(serial_lines),
      float(parse_errors),
      1.0 if stale else 0.0,
    ]
    self.pub_diag.publish(diag)

  def _on_reset_origin(self, _request, response):
    with self._lock:
      self._pitch_zero = self._pitch_raw
      self._roll_zero = self._roll_raw
      self._prev_rel_t = None
      self._prev_rel_pos = None
      self._prev_rel_vel = None
      self._t0 = time.monotonic()
    response.success = True
    response.message = 'Serial encoder origin reset'
    self.get_logger().info('/payload/encoder/reset_origin — serial origin reset')
    return response

  def destroy_node(self):
    self._stop.set()
    if self._ser is not None:
      try:
        self._ser.close()
      except Exception:
        pass
    super().destroy_node()


def main(args=None):
  rclpy.init(args=args)
  node: Optional[EncoderSerialNode] = None
  try:
    node = EncoderSerialNode()
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  except Exception as exc:
    print(f'[encoder_serial_node] Fatal: {exc}')
  finally:
    if node is not None:
      node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == '__main__':
  main()
