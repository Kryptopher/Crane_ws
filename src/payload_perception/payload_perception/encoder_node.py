#!/usr/bin/env python3
"""
encoder_node — ROS2 payload rope encoders (Jetson Orin Nano).

GPIO / quadrature decode matches ~/crane_ws/ops/encoder_reader.py:
  BOARD pins pitch 33/32 (GPIO13/GPIO07), roll 29/31 (GPIO01/GPIO11),
  optional GPIO debounce, state-gated quad table, IRQ on A+B when interrupt_on_b.

ROS extras: /payload/pose_e publishing, optional MOTION_START sync,
PPR-based degree scaling, invalid-transition diagnostics, thread-safe counts.

Do not run encoder_reader*.py while this node is up (GPIO busy).
"""

from __future__ import annotations

import os
import math
import threading
import time
from typing import Optional, Tuple

import rclpy
from gantry_control.msg import TrajCmd
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger

os.environ.setdefault('JETSON_MODEL_NAME', 'JETSON_ORIN_NANO')

_TRAJ_QOS = QoSProfile(
  reliability=ReliabilityPolicy.RELIABLE,
  durability=DurabilityPolicy.VOLATILE,
  depth=10,
)

# BOARD wiring — Pitch enc1, Roll enc2 (header GPIO names in comments)
DEFAULT_ENC1_A = 33   # Pitch A — GPIO13
DEFAULT_ENC1_B = 32   # Pitch B — GPIO07
DEFAULT_ENC2_A = 29   # Roll A  — GPIO01
DEFAULT_ENC2_B = 31   # Roll B  — GPIO11

QUAD_TABLE = {
  (0b00, 0b01): +1,
  (0b01, 0b11): +1,
  (0b11, 0b10): +1,
  (0b10, 0b00): +1,
  (0b00, 0b10): -1,
  (0b10, 0b11): -1,
  (0b11, 0b01): -1,
  (0b01, 0b00): -1,
}

POSE_E_FIELDS = ('time', 'pitch_deg', 'roll_deg', 'pitch_count', 'roll_count')
POSE_E_REL_FIELDS = (
  'time', 'pitch_deg', 'roll_deg',
  'x_rel_m', 'y_rel_m', 'z_rel_m',
  'vx_rel_m_s', 'vy_rel_m_s', 'vz_rel_m_s',
)
DIAG_FIELDS = (
  'time',
  'pitch_count', 'roll_count',
  'pitch_invalid', 'roll_invalid',
  'pitch_recovered', 'roll_recovered',
  'pitch_state', 'roll_state',
)


def quad_decode(
  last_state: int, a: int, b: int,
) -> Tuple[int, int, bool]:
  """
  encoder_reader_v2 stepping: only count on state change with valid delta.
  Returns (new_state, delta_counts, invalid_transition).
  """
  state = (a << 1) | b
  if state == last_state:
    return last_state, 0, False
  delta = QUAD_TABLE.get((last_state, state))
  if delta is None:
    return state, 0, True
  return state, delta, False


class EncoderNode(Node):
  def __init__(self):
    super().__init__('encoder_node')

    self.declare_parameter('publish_rate_hz', 50.0)
    self.declare_parameter('follow_sync_hz', True)
    self.declare_parameter('enc1_a', DEFAULT_ENC1_A)
    self.declare_parameter('enc1_b', DEFAULT_ENC1_B)
    self.declare_parameter('enc2_a', DEFAULT_ENC2_A)
    self.declare_parameter('enc2_b', DEFAULT_ENC2_B)
    self.declare_parameter('debounce_ms', 0)
    self.declare_parameter('use_pull_up', False)
    self.declare_parameter('interrupt_on_b', True)
    self.declare_parameter('decode_mode', 'x4')
    self.declare_parameter('recover_missed_edges', True)
    self.declare_parameter('zero_on_start', True)
    self.declare_parameter('wait_motion_start', False)
    self.declare_parameter('publish_when_idle', True)
    self.declare_parameter('counts_are_degrees', False)
    self.declare_parameter('degrees_per_count', 0.0)
    self.declare_parameter('ppr', 1000)
    self.declare_parameter('count_mode', 4)
    self.declare_parameter('gear_ratio', 1.0)
    self.declare_parameter('log_interval_sec', 0.0)
    self.declare_parameter('rope_length_m', 1.20)
    self.declare_parameter('rel_vel_ema_alpha', 0.35)
    self.declare_parameter('rel_vel_min_dt_s', 0.003)
    self.declare_parameter('rel_sign_x', 1.0)
    self.declare_parameter('rel_sign_y', 1.0)

    rate = float(self.get_parameter('publish_rate_hz').value)
    self._follow_sync_hz = bool(self.get_parameter('follow_sync_hz').value)
    self._publish_hz = max(5.0, min(rate, 120.0))
    self._wait_motion = bool(self.get_parameter('wait_motion_start').value)
    self._publish_when_idle = bool(self.get_parameter('publish_when_idle').value)
    self._motion_active = not self._wait_motion
    self.pin_A1 = int(self.get_parameter('enc1_a').value)
    self.pin_B1 = int(self.get_parameter('enc1_b').value)
    self.pin_A2 = int(self.get_parameter('enc2_a').value)
    self.pin_B2 = int(self.get_parameter('enc2_b').value)
    debounce_ms = int(self.get_parameter('debounce_ms').value)
    use_pull_up = bool(self.get_parameter('use_pull_up').value)
    self._interrupt_on_b = bool(self.get_parameter('interrupt_on_b').value)
    self._decode_mode = str(self.get_parameter('decode_mode').value).lower()
    if self._decode_mode not in ('x4', 'x2_a', 'x1_a'):
      self.get_logger().warn(
        f'Unsupported decode_mode={self._decode_mode!r}; using x4')
      self._decode_mode = 'x4'
    self._recover_missed_edges = bool(
      self.get_parameter('recover_missed_edges').value)
    self._counts_are_degrees = bool(self.get_parameter('counts_are_degrees').value)
    deg_override = float(self.get_parameter('degrees_per_count').value)
    log_interval = float(self.get_parameter('log_interval_sec').value)
    self._rope_length_m = max(0.05, float(self.get_parameter('rope_length_m').value))
    self._rel_vel_alpha = float(self.get_parameter('rel_vel_ema_alpha').value)
    self._rel_vel_alpha = max(0.0, min(self._rel_vel_alpha, 1.0))
    self._rel_vel_min_dt_s = max(1e-4, float(self.get_parameter('rel_vel_min_dt_s').value))
    self._rel_sign_x = 1.0 if float(self.get_parameter('rel_sign_x').value) >= 0.0 else -1.0
    self._rel_sign_y = 1.0 if float(self.get_parameter('rel_sign_y').value) >= 0.0 else -1.0

    ppr = int(self.get_parameter('ppr').value)
    count_mode = int(self.get_parameter('count_mode').value)
    if self._decode_mode == 'x2_a' and count_mode == 4:
      count_mode = 2
    elif self._decode_mode == 'x1_a' and count_mode in (2, 4):
      count_mode = 1
    gear_ratio = float(self.get_parameter('gear_ratio').value)
    counts_per_rev = max(1, ppr * count_mode * gear_ratio)
    self._deg_per_count = (
      deg_override if deg_override > 0.0
      else 360.0 / counts_per_rev
    )

    self._count_lock = threading.Lock()
    self.pitch_count = 0
    self.roll_count = 0
    self.last_pitch_state = 0b00
    self.last_roll_state = 0b00
    self.error_count = {'pitch': 0, 'roll': 0}
    self.recovered_count = {'pitch': 0, 'roll': 0}
    self._last_delta = {'pitch': 0, 'roll': 0}
    self._t0 = time.monotonic()
    self._last_log_t = 0.0
    self._log_interval = log_interval if log_interval > 0.0 else 0.0
    self._prev_rel_t: Optional[float] = None
    self._prev_rel_pos: Optional[Tuple[float, float, float]] = None
    self._prev_rel_vel: Optional[Tuple[float, float, float]] = None

    self.pub = self.create_publisher(
      Float64MultiArray, '/payload/pose_e', qos_profile_sensor_data)
    self.pub_rel = self.create_publisher(
      Float64MultiArray, '/payload/pose_e_rel', qos_profile_sensor_data)
    self.pub_diag = self.create_publisher(
      Float64MultiArray, '/payload/encoder/diagnostics', qos_profile_sensor_data)
    self.create_service(
      Trigger, '/payload/encoder/reset_origin', self._on_reset_origin)
    self.add_on_set_parameters_callback(self._on_set_parameters)

    import Jetson.GPIO as GPIO
    self._GPIO = GPIO
    GPIO.setwarnings(False)
    try:
      GPIO.cleanup()
    except Exception:
      pass
    GPIO.setmode(GPIO.BOARD)

    # Jetson.GPIO ignores pull_up_down; v2 script requests PUD_UP on hardware that supports it
    for pin in (self.pin_A1, self.pin_B1, self.pin_A2, self.pin_B2):
      if use_pull_up:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
      else:
        GPIO.setup(pin, GPIO.IN)

    self.last_pitch_state = (
      (GPIO.input(self.pin_A1) << 1) | GPIO.input(self.pin_B1))
    self.last_roll_state = (
      (GPIO.input(self.pin_A2) << 1) | GPIO.input(self.pin_B2))

    self._irq_pins = [self.pin_A1, self.pin_A2]
    if self._decode_mode == 'x4' and self._interrupt_on_b:
      self._irq_pins.extend([self.pin_B1, self.pin_B2])

    for pin in self._irq_pins:
      cb = self._pitch_cb if pin in (self.pin_A1, self.pin_B1) else self._roll_cb
      if debounce_ms > 0:
        GPIO.add_event_detect(
          pin, GPIO.BOTH, callback=cb, bouncetime=debounce_ms)
      else:
        GPIO.add_event_detect(pin, GPIO.BOTH, callback=cb)

    if self.get_parameter('zero_on_start').value:
      self.reset_counts()

    self._publish_timer = self.create_timer(
      1.0 / self._publish_hz, self._publish)
    self.get_logger().info(
      f'Encoders ready pitch BOARD ({self.pin_A1},{self.pin_B1}) '
      f'GPIO13/GPIO07 roll BOARD ({self.pin_A2},{self.pin_B2}) '
      f'GPIO01/GPIO11 mode={self._decode_mode} irq_b={self._interrupt_on_b} '
      f'debounce={max(0, debounce_ms)}ms recover_missed={self._recover_missed_edges} '
      f'@ {self._publish_hz:.1f} Hz → /payload/pose_e + /payload/pose_e_rel '
      f'(L={self._rope_length_m:.2f} m)')
    if self._follow_sync_hz:
      self.create_subscription(
        Float64, '/stack/pose_sync_hz', self._on_pose_sync_hz, 10)
      self.get_logger().info(
        'follow_sync_hz: retiming /payload/pose_e from /stack/pose_sync_hz')
    if self._wait_motion and not self._publish_when_idle:
      self.get_logger().info(
        'Publishing /payload/pose_e only after /traj_cmd MOTION_START '
        '(set publish_when_idle:=true for dashboard)')
    elif self._wait_motion:
      self.get_logger().info(
        'wait_motion_start=true — t=0 at MOTION_START; publishing idle samples until then')
    if self._wait_motion:
      self.create_subscription(
        TrajCmd, '/traj_cmd', self._on_traj_cmd, _TRAJ_QOS)

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

  def _counts_to_degrees(self, counts: int) -> float:
    if self._counts_are_degrees:
      return float(counts)
    return float(counts) * self._deg_per_count

  def _on_traj_cmd(self, msg: TrajCmd):
    if msg.command != TrajCmd.MOTION_START:
      return
    self._motion_active = True
    self._t0 = time.monotonic()
    if self.get_parameter('zero_on_start').value:
      self.reset_counts()
    self._prev_rel_t = None
    self._prev_rel_pos = None
    self._prev_rel_vel = None
    self.get_logger().info('MOTION_START — encoder clock reset')

  def _step_axis(
    self,
    *,
    is_pitch: bool,
    a: int,
    b: int,
  ) -> None:
    with self._count_lock:
      if is_pitch:
        if self._decode_mode == 'x1_a':
          self.last_pitch_state = (a << 1) | b
          if a:
            delta = 1 if a == b else -1
            self.pitch_count += delta
            self._last_delta['pitch'] = delta
          return
        if self._decode_mode == 'x2_a':
          delta = 1 if a == b else -1
          self.pitch_count += delta
          self._last_delta['pitch'] = delta
          self.last_pitch_state = (a << 1) | b
          return
        last = self.last_pitch_state
        new_state, delta, bad = quad_decode(last, a, b)
        axis = 'pitch'
        if bad:
          self.error_count[axis] += 1
          if self._recover_missed_edges and self._last_delta[axis] != 0:
            delta = 2 * self._last_delta[axis]
            self.recovered_count[axis] += 1
        if delta:
          self.pitch_count += delta
          self._last_delta[axis] = 1 if delta > 0 else -1
        self.last_pitch_state = new_state
      else:
        if self._decode_mode == 'x1_a':
          self.last_roll_state = (a << 1) | b
          if a:
            delta = 1 if a == b else -1
            self.roll_count += delta
            self._last_delta['roll'] = delta
          return
        if self._decode_mode == 'x2_a':
          delta = 1 if a == b else -1
          self.roll_count += delta
          self._last_delta['roll'] = delta
          self.last_roll_state = (a << 1) | b
          return
        last = self.last_roll_state
        new_state, delta, bad = quad_decode(last, a, b)
        axis = 'roll'
        if bad:
          self.error_count[axis] += 1
          if self._recover_missed_edges and self._last_delta[axis] != 0:
            delta = 2 * self._last_delta[axis]
            self.recovered_count[axis] += 1
        if delta:
          self.roll_count += delta
          self._last_delta[axis] = 1 if delta > 0 else -1
        self.last_roll_state = new_state

  def _pitch_cb(self, _channel):
    a = self._GPIO.input(self.pin_A1)
    b = self._GPIO.input(self.pin_B1)
    self._step_axis(is_pitch=True, a=a, b=b)

  def _roll_cb(self, _channel):
    a = self._GPIO.input(self.pin_A2)
    b = self._GPIO.input(self.pin_B2)
    self._step_axis(is_pitch=False, a=a, b=b)

  def _set_publish_hz(self, hz: float) -> None:
    hz = max(5.0, min(float(hz), 120.0))
    if abs(hz - self._publish_hz) < 0.5:
      return
    self._publish_hz = hz
    if self._publish_timer is not None:
      self._publish_timer.cancel()
    self._publish_timer = self.create_timer(1.0 / hz, self._publish)
    self.get_logger().info(
      f'Following /stack/pose_sync_hz → /payload/pose_e @ {hz:.1f} Hz')

  def _on_pose_sync_hz(self, msg: Float64) -> None:
    if not self._follow_sync_hz:
      return
    self._set_publish_hz(msg.data)

  def _publish(self):
    if self._wait_motion and not self._motion_active and not self._publish_when_idle:
      return
    with self._count_lock:
      pitch = self.pitch_count
      roll = self.roll_count
      err_p = self.error_count['pitch']
      err_r = self.error_count['roll']
      rec_p = self.recovered_count['pitch']
      rec_r = self.recovered_count['roll']
      state_p = self.last_pitch_state
      state_r = self.last_roll_state
    if self._wait_motion and not self._motion_active:
      t = 0.0
    else:
      t = time.monotonic() - self._t0
    msg = Float64MultiArray()
    dim = MultiArrayDimension()
    dim.label = ','.join(POSE_E_FIELDS)
    dim.size = 5
    dim.stride = 5
    msg.layout.dim.append(dim)
    msg.data = [
      float(t),
      self._counts_to_degrees(pitch),
      self._counts_to_degrees(roll),
      float(pitch),
      float(roll),
    ]
    self.pub.publish(msg)
    rel_msg = self._build_rel_msg(t, msg.data[1], msg.data[2])
    self.pub_rel.publish(rel_msg)
    self.pub_diag.publish(
      self._build_diag_msg(t, pitch, roll, err_p, err_r, rec_p, rec_r, state_p, state_r))

    if self._log_interval > 0.0:
      now = time.monotonic()
      if now - self._last_log_t >= self._log_interval:
        self._last_log_t = now
        self.get_logger().debug(
          f'pose_e pitch={msg.data[1]:+.2f}° roll={msg.data[2]:+.2f}° '
          f'counts {pitch:+d}/{roll:+d} err {err_p}/{err_r} '
          f'recovered {rec_p}/{rec_r}')

  def _payload_relative_from_angles(
    self, pitch_deg: float, roll_deg: float
  ) -> Tuple[float, float, float]:
    """Payload offset from cart attach point in the gantry frame.

    Convention (confirmed on rig): pitch -> gantry X,
    roll -> gantry Y, both 0 at rest. Straight-down rest is
    (0, 0, -L); Z is negative downward. rel_sign_x/rel_sign_y flip the
    horizontal swing direction to match physical encoder polarity.
    """
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

  def _build_diag_msg(
    self,
    t: float,
    pitch: int,
    roll: int,
    err_p: int,
    err_r: int,
    rec_p: int,
    rec_r: int,
    state_p: int,
    state_r: int,
  ) -> Float64MultiArray:
    msg = Float64MultiArray()
    dim = MultiArrayDimension()
    dim.label = ','.join(DIAG_FIELDS)
    dim.size = len(DIAG_FIELDS)
    dim.stride = len(DIAG_FIELDS)
    msg.layout.dim.append(dim)
    msg.data = [
      float(t),
      float(pitch), float(roll),
      float(err_p), float(err_r),
      float(rec_p), float(rec_r),
      float(state_p), float(state_r),
    ]
    return msg

  def reset_counts(self):
    with self._count_lock:
      self.pitch_count = 0
      self.roll_count = 0
      self.error_count = {'pitch': 0, 'roll': 0}
      self.recovered_count = {'pitch': 0, 'roll': 0}
      self._last_delta = {'pitch': 0, 'roll': 0}
      if hasattr(self, '_GPIO'):
        self.last_pitch_state = (
          (self._GPIO.input(self.pin_A1) << 1) | self._GPIO.input(self.pin_B1))
        self.last_roll_state = (
          (self._GPIO.input(self.pin_A2) << 1) | self._GPIO.input(self.pin_B2))
      self._prev_rel_t = None
      self._prev_rel_pos = None
      self._prev_rel_vel = None
    self.get_logger().debug('Encoder counts reset')

  def _on_reset_origin(self, _request, response):
    self.reset_counts()
    self._t0 = time.monotonic()
    response.success = True
    response.message = 'Encoder origin reset'
    self.get_logger().info('/payload/encoder/reset_origin — encoder origin reset')
    return response

  def destroy_node(self):
    GPIO = self._GPIO
    for pin in self._irq_pins:
      try:
        GPIO.remove_event_detect(pin)
      except Exception:
        pass
    try:
      GPIO.cleanup()
    except Exception:
      pass
    self.get_logger().info(
      f'Encoder stopped — errors pitch={self.error_count["pitch"]} '
      f'roll={self.error_count["roll"]}')
    super().destroy_node()


def main(args=None):
  rclpy.init(args=args)
  node: Optional[EncoderNode] = None
  try:
    node = EncoderNode()
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  except Exception as exc:
    print(f'[encoder_node] Fatal: {exc}')
    if node is None:
      try:
        import Jetson.GPIO as GPIO
        GPIO.cleanup()
      except Exception:
        pass
  finally:
    if node is not None:
      node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == '__main__':
  main()
