#!/usr/bin/env python3
"""
encoder_serial_node — Arduino Nano payload encoder bridge.

pitch_count/roll_count are the 21-bit magnetic encoders' absolute shaft
readings, so they wrap: the firmware reports them sign-extended into
[-counts_per_rev/2, +counts_per_rev/2), and a swing across that boundary makes
the reported count step by a full revolution. This node unwraps them (see
`unwrap_counts`) before applying its own software origin and rescaling to
degrees, so everything published downstream is continuous.

Reads lines from USB serial, either the original encoder-only format:
  millis,pitch_count,roll_count

or the current encoder+gyro(dps)+radio-status format:
  millis,pitch_count,roll_count,gx1_dps,gy1_dps,gz1_dps,gx2_dps,gy2_dps,gz2_dps,packet_age_ms,packet_seen

or that same format with two trailing CRC-fail counters (not currently
surfaced downstream, just tolerated so the line still parses):
  millis,pitch_count,roll_count,gx1_dps,gy1_dps,gz1_dps,gx2_dps,gy2_dps,gz2_dps,packet_age_ms,packet_seen,pitch_crc_fails,roll_crc_fails

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
  'vx_rel_raw_m_s', 'vy_rel_raw_m_s', 'vz_rel_raw_m_s',
  'sample_age_ms',
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
  'wrap_events',
  'sample_age_ms',
)


class EncoderSerialNode(Node):
  def __init__(self):
    super().__init__('encoder_serial_node')

    self.declare_parameter('serial_port', '/dev/ttyACM0')
    self.declare_parameter('baud', 500000)
    self.declare_parameter('publish_rate_hz', 100.0)
    self.declare_parameter('zero_on_start', True)
    self.declare_parameter('counts_per_rev', 2097152)
    self.declare_parameter('unwrap_counts', True)
    self.declare_parameter('invert_pitch', False)
    self.declare_parameter('invert_roll', False)
    self.declare_parameter('rope_length_m', 1.20)
    self.declare_parameter('rel_vel_ema_alpha', 0.35)
    self.declare_parameter('rel_vel_min_dt_s', 0.003)
    self.declare_parameter('rel_sign_x', 1.0)
    self.declare_parameter('rel_sign_y', 1.0)
    self.declare_parameter('stale_timeout_s', 0.5)

    self._configured_port = str(self.get_parameter('serial_port').value)
    # Provisional; _open_serial_with_retry() picks and verifies the real port.
    self._port = self._configured_port
    self._baud = int(self.get_parameter('baud').value)
    publish_hz = float(self.get_parameter('publish_rate_hz').value)
    self._publish_hz = max(5.0, min(publish_hz, 200.0))
    self._zero_on_start = bool(self.get_parameter('zero_on_start').value)
    self._counts_per_rev = max(1, int(self.get_parameter('counts_per_rev').value))
    self._deg_per_count = 360.0 / self._counts_per_rev
    self._unwrap_counts = (
      bool(self.get_parameter('unwrap_counts').value) and self._counts_per_rev > 1)
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
    # Last count exactly as the firmware reported it (still wrapped), used to
    # accumulate self._pitch_raw / self._roll_raw as continuous values.
    self._pitch_wrapped: Optional[int] = None
    self._roll_wrapped: Optional[int] = None
    self._wrap_events = 0
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
    self._last_raw_rel_vel: Optional[Tuple[float, float, float]] = None
    self._last_published_serial_lines = -1
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
      f'@ {self._publish_hz:.1f} Hz deg/count={self._deg_per_count:.6f} '
      f'({self._counts_per_rev} counts/rev, '
      f'unwrap={"on" if self._unwrap_counts else "off"}, '
      f'L={self._rope_length_m:.2f} m)')

  @staticmethod
  def _looks_like_encoder_line(line: str) -> bool:
    """True if a serial line has the encoder CSV shape (millis + counts + ...)."""
    line = line.strip()
    if not line or line.lower().startswith('time_ms,'):
      return False
    parts = line.split(',')
    if len(parts) not in (3, 11, 13, 15, 17):
      return False
    try:
      float(parts[0])
      float(parts[1])
      float(parts[2])
    except ValueError:
      return False
    return True

  def _port_speaks_encoder(self, ser, probe_s: float = 1.5) -> bool:
    """Read briefly and check the stream is the encoder protocol, not another
    Arduino on an identically-named CH340 adapter (e.g. the ic_cart board)."""
    try:
      ser.reset_input_buffer()
    except Exception:
      pass
    deadline = time.monotonic() + probe_s
    saw_any = False
    while time.monotonic() < deadline:
      try:
        raw = ser.readline()
      except Exception:
        return False
      if not raw:
        continue
      saw_any = True
      if self._looks_like_encoder_line(raw.decode('ascii', errors='ignore')):
        return True
    # No traffic at all → can't tell; let the caller keep this port and retry.
    return not saw_any

  def _candidate_ports(self) -> list:
    if os.path.exists(self._configured_port):
      ordered = [self._configured_port]
    else:
      ordered = []
    for pattern in (
      '/dev/serial/by-id/*', '/dev/serial/by-path/*',
      '/dev/ttyCH341USB*', '/dev/ttyUSB*', '/dev/ttyACM*',
    ):
      for path in sorted(glob.glob(pattern)):
        if path not in ordered and os.path.exists(path):
          ordered.append(path)
    return ordered

  def _open_serial_with_retry(self):
    deadline = time.monotonic() + 12.0
    last_exc: Exception | None = None
    last_port = self._configured_port
    while time.monotonic() < deadline:
      candidates = self._candidate_ports()
      if not candidates:
        self.get_logger().warn(
          f'No serial ports match {self._configured_port}; waiting')
        time.sleep(0.5)
        continue
      for port in candidates:
        last_port = port
        try:
          ser = serial.Serial(port, self._baud, timeout=0.1)
        except Exception as exc:
          last_exc = exc
          continue
        # Reject a wrong-but-openable device (a second CH340 running some other
        # firmware) unless it is the only thing we can try.
        if len(candidates) > 1 and not self._port_speaks_encoder(ser):
          self.get_logger().warn(
            f'{port} opened but did not produce encoder-format lines; '
            'trying the next candidate')
          ser.close()
          continue
        if port != self._configured_port:
          self.get_logger().warn(
            f'Configured serial port {self._configured_port} unavailable; '
            f'using {port} (verified encoder stream)')
        self._port = port
        return ser
      time.sleep(0.5)

    if last_exc is not None:
      raise RuntimeError(
        f'could not open encoder serial port {self._configured_port} '
        f'(last tried {last_port}) after 12s: {last_exc}')
    raise RuntimeError(
      f'could not find an encoder serial port matching {self._configured_port} '
      'after 12s')

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
        if len(parts) not in (3, 11, 13, 15, 17):
          raise ValueError('expected 3, 11, 13, 15, or 17 comma-separated fields')
        arduino_ms = float(parts[0])
        pitch_raw = int(float(parts[1]))
        roll_raw = int(float(parts[2]))
        if len(parts) in (11, 13):
          # 13-field format appends pitch_crc_fails,roll_crc_fails, which
          # this node doesn't currently surface downstream.
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
        self._pitch_raw, self._roll_raw = self._unwrap_locked(pitch_raw, roll_raw)
        self._pitch_wrapped = pitch_raw
        self._roll_wrapped = roll_raw
        self._gyro_dps = gyro_dps
        self._packet_age_ms = packet_age_ms
        self._packet_seen = packet_seen
        if self._zero_on_start and self._pitch_zero is None:
          self._pitch_zero = self._pitch_raw
          self._roll_zero = self._roll_raw
        self._last_rx_mono = time.monotonic()
        self._serial_lines += 1

  def _unwrap_locked(self, pitch_wrapped: int, roll_wrapped: int) -> Tuple[int, int]:
    """Accumulate wrapped absolute counts into continuous counts.

    The 21-bit encoders report an absolute shaft angle, so the count jumps by a
    full revolution whenever the payload swings across the firmware's wrap
    point. Folding each sample-to-sample delta back into
    (-counts_per_rev/2, +counts_per_rev/2] removes those jumps. At the ~100 Hz
    serial rate a genuine half-revolution between samples would be 18000 deg/s,
    so no real payload motion is misread as a wrap.
    """
    if not self._unwrap_counts:
      return pitch_wrapped, roll_wrapped

    cpr = self._counts_per_rev
    half = cpr // 2
    if self._pitch_wrapped is None or self._roll_wrapped is None:
      return pitch_wrapped, roll_wrapped

    d_pitch = ((pitch_wrapped - self._pitch_wrapped + half) % cpr) - half
    d_roll = ((roll_wrapped - self._roll_wrapped + half) % cpr) - half
    if abs(pitch_wrapped - self._pitch_wrapped) > half:
      self._wrap_events += 1
    if abs(roll_wrapped - self._roll_wrapped) > half:
      self._wrap_events += 1
    return self._pitch_raw + d_pitch, self._roll_raw + d_roll

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
  ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if self._prev_rel_t is None or self._prev_rel_pos is None:
      self._prev_rel_t = t
      self._prev_rel_pos = pos
      nan_velocity = (float('nan'), float('nan'), float('nan'))
      self._last_raw_rel_vel = nan_velocity
      return nan_velocity, nan_velocity
    dt = t - self._prev_rel_t
    if dt < self._rel_vel_min_dt_s:
      filtered = self._prev_rel_vel or (
        float('nan'), float('nan'), float('nan'))
      raw = self._last_raw_rel_vel or filtered
      return filtered, raw
    raw = (
      (pos[0] - self._prev_rel_pos[0]) / dt,
      (pos[1] - self._prev_rel_pos[1]) / dt,
      (pos[2] - self._prev_rel_pos[2]) / dt,
    )
    vx, vy, vz = raw
    if self._prev_rel_vel is not None and self._rel_vel_alpha > 0.0:
      a = self._rel_vel_alpha
      vx = a * vx + (1.0 - a) * self._prev_rel_vel[0]
      vy = a * vy + (1.0 - a) * self._prev_rel_vel[1]
      vz = a * vz + (1.0 - a) * self._prev_rel_vel[2]
    self._prev_rel_t = t
    self._prev_rel_pos = pos
    self._prev_rel_vel = (vx, vy, vz)
    self._last_raw_rel_vel = raw
    return (vx, vy, vz), raw

  def _build_rel_msg(
    self,
    t: float,
    pitch_deg: float,
    roll_deg: float,
    sample_age_ms: float,
  ) -> Float64MultiArray:
    x_rel, y_rel, z_rel = self._payload_relative_from_angles(pitch_deg, roll_deg)
    filtered_velocity, raw_velocity = self._payload_relative_velocity(
      t, (x_rel, y_rel, z_rel))
    vx_rel, vy_rel, vz_rel = filtered_velocity
    vx_raw, vy_raw, vz_raw = raw_velocity
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
      float(vx_raw), float(vy_raw), float(vz_raw),
      float(sample_age_ms),
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
      wrap_events = self._wrap_events
      stale = self._last_rx_mono <= 0.0 or now - self._last_rx_mono > self._stale_timeout_s
      sample_mono = self._last_rx_mono
      new_sample = serial_lines != self._last_published_serial_lines
      if new_sample:
        self._last_published_serial_lines = serial_lines

    if stale:
      # Never republish a frozen IMU sample as if it were live — downstream
      # loggers key off message arrival, not on serial health, so a dead link
      # would otherwise be recorded as a long run of valid constant readings.
      gyro_dps = [float('nan')] * 6
      packet_age_ms = float('nan')
      packet_seen = float('nan')

    # The counts belong to the most recently received serial packet.  Stamp
    # them at serial receipt, rather than at this independent publication
    # timer, so downstream phase prediction does not interpret an old sample
    # as a current one.  At 500 kbaud the remaining wire-time bias is a few
    # milliseconds, versus the observed 10--25 ms timer/queue error.
    t = sample_mono - self._t0
    sample_age_ms = 1000.0 * max(0.0, now - sample_mono)
    pitch_deg = self._counts_to_degrees(pitch)
    roll_deg = self._counts_to_degrees(roll)

    # Do not make a disconnected serial encoder look healthy by republishing
    # its last position with a fresh ROS receive time.  Downstream motion nodes
    # use pose-message freshness as a safety gate, so frozen samples must stop.
    if not stale and new_sample:
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
      self.pub_rel.publish(
        self._build_rel_msg(t, pitch_deg, roll_deg, sample_age_ms))
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
      float(wrap_events),
      float(sample_age_ms),
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
