#!/usr/bin/env python3
"""
Crane ops dashboard — HTTP server for crane_ops.html + ROS2 bridge.

Serves the mission planner UI and exposes gantry state, payload encoders,
and waypoint missions. Vision is ROS topics only (no MJPEG :8080 proxy).

  ros2 run gantry_control crane_dashboard_server.py
  # or: ros2 launch gantry_control mission_planner.launch.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Empty, Float64, Float64MultiArray, String
from std_srvs.srv import Trigger

from gantry_control.msg import GantryState
from gantry_control.srv import IcCartMove, MoveTo, SetMode
from payload_perception_msgs.msg import PayloadState


class CraneDashboardServer(Node):
    def __init__(self) -> None:
        super().__init__('crane_dashboard_server')
        self.declare_parameter('dashboard_port', 8081)
        self.declare_parameter('camera_proxy', False)
        self.declare_parameter('stream_port', 8080)
        self.declare_parameter('workspace_m', 1.00)
        self.declare_parameter('move_timeout_s', 180.0)
        self.declare_parameter('position_tolerance_m', 0.008)
        self.declare_parameter('homing_timeout_s', 45.0)
        self.declare_parameter('stack_pose_publish_hz', 100.0)
        self.declare_parameter('csv_dir', '/home/sanjay/crane_ws/csv')
        self.declare_parameter('csv_log_dir', '~/payload_logs')
        self.declare_parameter('csv_log_hz', 50.0)

        self.dashboard_port = int(self.get_parameter('dashboard_port').value)
        self.camera_proxy = bool(self.get_parameter('camera_proxy').value)
        self.stream_port = int(self.get_parameter('stream_port').value)
        self.workspace_m = float(self.get_parameter('workspace_m').value)
        self.move_timeout_s = float(self.get_parameter('move_timeout_s').value)
        self.position_tolerance_m = float(self.get_parameter('position_tolerance_m').value)
        self.homing_timeout_s = float(self.get_parameter('homing_timeout_s').value)
        self.stack_pose_publish_hz = float(
            self.get_parameter('stack_pose_publish_hz').value)
        self.csv_dir = os.path.expanduser(str(self.get_parameter('csv_dir').value))
        self.csv_log_dir = os.path.expanduser(str(self.get_parameter('csv_log_dir').value))
        self.csv_log_hz = float(self.get_parameter('csv_log_hz').value)
        self._stack_sync_hz: float = 0.0
        self._stack_sync_last_rx: float = 0.0

        self._cb = ReentrantCallbackGroup()
        self._gantry_lock = threading.Lock()
        self._media_lock = threading.Lock()

        self.gantry: Dict[str, Any] = {
            'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0,
            'mode': 'IDLE', 'enabled': False, 'homed': False,
            'homing_active': False, 'homing_status': '',
            'estop': False, 'move_done': True,
        }
        self._prev_homed: bool = False
        self.pose_e: Optional[Dict[str, float]] = None
        self.pose_e_rel: Optional[Dict[str, float]] = None
        self.payload: Optional[Dict[str, Any]] = None
        self._camera_jpeg: Optional[bytes] = None
        self.waypoints: List[Dict[str, Any]] = []
        self.mission_running = False
        self._mission_stop = threading.Event()
        self._motion_lock = threading.Lock()
        self._goto_running = False
        self._goto_status: Dict[str, str] = {'state': 'idle', 'message': ''}
        self._stabilizer_proc: subprocess.Popen | None = None
        self._gantry_last_rx: float = 0.0
        self._payload_last_rx: float = 0.0
        self._pose_e_last_rx: float = 0.0
        self._pose_e_rel_last_rx: float = 0.0
        self._camera_last_rx: float = 0.0
        self.ic_cart: Optional[Dict[str, float]] = None
        self._ic_cart_last_rx: float = 0.0
        self._gantry_stale_s: float = 30.0
        self._gantry_loop_rate_hz: float = 0.0
        self._gantry_rx_times: deque = deque(maxlen=120)
        self._payload_rx_times: deque = deque(maxlen=120)
        self._pose_e_rx_times: deque = deque(maxlen=120)
        self._pose_e_rel_rx_times: deque = deque(maxlen=120)
        self._rate_window_s: float = 2.0
        self._csv_log_lock = threading.Lock()
        self._csv_log_fp = None
        self._csv_log_writer = None
        self._csv_log_path = ''
        self._csv_log_planned_path = ''
        self._csv_log_rows = 0
        self._csv_log_started = 0.0
        self.create_timer(
            1.0 / max(1.0, self.csv_log_hz),
            self._write_csv_log_sample,
            callback_group=self._cb)

        self.create_subscription(
            GantryState, '/gantry/state', self._on_gantry_state, 10,
            callback_group=self._cb)
        self.create_subscription(
            Float64MultiArray, '/payload/pose_e', self._on_pose_e,
            qos_profile_sensor_data,
            callback_group=self._cb)
        self.create_subscription(
            Float64MultiArray, '/payload/pose_e_rel', self._on_pose_e_rel,
            qos_profile_sensor_data,
            callback_group=self._cb)
        self.create_subscription(
            Float64MultiArray, '/ic_cart/state', self._on_ic_cart_state,
            qos_profile_sensor_data,
            callback_group=self._cb)
        self.create_subscription(
            PayloadState, '/payload/state', self._on_payload_state,
            qos_profile_sensor_data,
            callback_group=self._cb)
        self.create_subscription(
            CompressedImage, '/payload/camera/compressed', self._on_camera,
            qos_profile_sensor_data,
            callback_group=self._cb)
        self.create_subscription(
            Float64, '/stack/pose_sync_hz', self._on_stack_sync_hz, 10,
            callback_group=self._cb)

        self._enable_cli = self.create_client(Trigger, '/gantry/enable')
        self._disable_cli = self.create_client(Trigger, '/gantry/disable')
        self._estop_cli = self.create_client(Trigger, '/gantry/estop')
        self._clear_estop_cli = self.create_client(Trigger, '/gantry/clear_estop')
        self._set_mode_cli = self.create_client(SetMode, '/gantry/set_mode')
        self._move_to_cli = self.create_client(MoveTo, '/gantry/move_to')
        self._encoder_reset_cli = self.create_client(
            Trigger, '/payload/encoder/reset_origin')
        self._encoder_serial_params_cli = self.create_client(
            SetParameters, '/encoder_serial_node/set_parameters')
        self._encoder_gpio_params_cli = self.create_client(
            SetParameters, '/encoder_node/set_parameters')
        self._gantry_params_cli = self.create_client(
            SetParameters, '/gantry_controller/set_parameters')
        self._gantry_get_params_cli = self.create_client(
            GetParameters, '/gantry_controller/get_parameters')
        self._ic_cart_move_cli = self.create_client(
            IcCartMove, '/ic_cart/move_to')
        self._ic_cart_cal_cli = self.create_client(
            Trigger, '/ic_cart/calibrate_origin')
        self._ic_cart_stop_cli = self.create_client(Trigger, '/ic_cart/stop')
        self._ic_cart_enable_cli = self.create_client(Trigger, '/ic_cart/enable')
        self._ic_cart_disable_cli = self.create_client(
            Trigger, '/ic_cart/disable')
        self._reset_vision_pub = self.create_publisher(
            Empty, '/payload/reset_origin', 10)
        self._exp_start_pub = self.create_publisher(
            Empty, '/experiment/start', 10)
        self._exp_end_pub = self.create_publisher(
            Empty, '/experiment/end', 10)
        self._exp_start_x_pub = self.create_publisher(
            Float64, '/experiment/set_start_x', 10)
        self._exp_stop_pub = self.create_publisher(
            Empty, '/experiment/stop', 10)
        self._exp_save_pub = self.create_publisher(
            String, '/experiment/save', 10)
        self._experiment_state = 'idle'
        self._experiment_run_active = False

        # ── Camera↔encoder drift tracking ──
        # The encoder is the source of truth; the camera is expected to
        # disagree (small tags).  Deltas are only sampled when the system is
        # quiescent (move done, swing settled, camera tracking) so real payload
        # swing is never counted as drift.  Displayed, never alarmed on.
        self._drift_ewma_tau_s = 3.0
        self._drift_swing_settle_m_s = 0.01
        self._drift: Dict[str, Any] = {}
        self._drift_history: deque = deque(maxlen=300)
        self._drift_last_hist_t: float = 0.0
        self._reset_drift_stats()

        share = self._share_dir()
        self.web_dir = os.path.join(share, 'web')
        self.get_logger().info(
            f'Dashboard http://0.0.0.0:{self.dashboard_port}/')

    @staticmethod
    def _share_dir() -> str:
        try:
            from ament_index_python.packages import get_package_share_directory
            return get_package_share_directory('gantry_control')
        except Exception:
            return os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..'))

    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        """Replace NaN/Inf so /state is valid JSON for browser JSON.parse."""
        if isinstance(obj, dict):
            return {k: CraneDashboardServer._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [CraneDashboardServer._sanitize_for_json(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    @staticmethod
    def _estimate_hz(times: deque, now: float, window_s: float) -> float:
        samples = tuple(times)
        if len(samples) < 2:
            return 0.0
        recent = [t for t in samples if now - t <= window_s]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        if span < 1e-6:
            return 0.0
        return (len(recent) - 1) / span

    def _list_csv_profiles(self) -> List[Dict[str, str]]:
        try:
            entries = []
            for name in sorted(os.listdir(self.csv_dir)):
                if not name.lower().endswith('.csv'):
                    continue
                path = os.path.join(self.csv_dir, name)
                if os.path.isfile(path):
                    entries.append({'name': name, 'path': path})
            return entries
        except OSError:
            return []

    def _next_csv_log_path(self) -> str:
        os.makedirs(self.csv_log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.csv_log_dir, f'csv_mode_{ts}.csv')

    def _csv_log_state(self) -> Dict[str, Any]:
        with self._csv_log_lock:
            active = self._csv_log_writer is not None
            path = self._csv_log_path
            if not active and not self._csv_log_planned_path:
                self._csv_log_planned_path = self._next_csv_log_path()
            planned_path = self._csv_log_planned_path
            rows = self._csv_log_rows
        return {
            'active': active,
            'path': path or planned_path,
            'rows': rows,
            'hz': self.csv_log_hz,
        }

    def _start_csv_log(self) -> tuple[bool, str]:
        with self._csv_log_lock:
            if self._csv_log_writer is not None:
                return True, f'CSV logger already running: {self._csv_log_path}'
            path = self._csv_log_planned_path or self._next_csv_log_path()
            try:
                self._csv_log_fp = open(path, 'w', newline='')
            except OSError as exc:
                return False, f'Cannot open CSV log: {exc}'
            self._csv_log_writer = csv.writer(self._csv_log_fp)
            self._csv_log_path = path
            self._csv_log_planned_path = ''
            self._csv_log_rows = 0
            self._csv_log_started = time.monotonic()
            self._csv_log_writer.writerow([
                'time_sec',
                'cart_x_m', 'cart_y_m', 'cart_vx_m_s', 'cart_vy_m_s',
                'mode', 'enabled', 'homed',
                'pitch_deg', 'roll_deg', 'pitch_count', 'roll_count',
                'x_rel_m', 'y_rel_m', 'z_rel_m',
                'vx_rel_m_s', 'vy_rel_m_s', 'vz_rel_m_s',
            ])
            self._csv_log_fp.flush()
        self.get_logger().info(f'CSV mode log started → {path}')
        return True, f'CSV log started: {path}'

    def _stop_csv_log(self) -> tuple[bool, str]:
        with self._csv_log_lock:
            if self._csv_log_writer is None:
                return True, 'CSV logger is not running'
            path = self._csv_log_path
            rows = self._csv_log_rows
            fp = self._csv_log_fp
            self._csv_log_fp = None
            self._csv_log_writer = None
            self._csv_log_path = ''
            self._csv_log_planned_path = self._next_csv_log_path()
            self._csv_log_rows = 0
            self._csv_log_started = 0.0
        if fp is not None:
            fp.flush()
            fp.close()
        self.get_logger().info(f'CSV mode log stopped: {rows} rows → {path}')
        return True, f'CSV log stopped: {rows} rows → {path}'

    def _write_csv_log_sample(self) -> None:
        with self._csv_log_lock:
            writer = self._csv_log_writer
            fp = self._csv_log_fp
            started = self._csv_log_started
        if writer is None or fp is None:
            return

        with self._gantry_lock:
            g = dict(self.gantry)
            pose_e = dict(self.pose_e) if self.pose_e else {}
            pose_e_rel = dict(self.pose_e_rel) if self.pose_e_rel else {}

        def fval(data: Dict[str, Any], key: str) -> str:
            val = data.get(key, None)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                return f'{float(val):.6f}'
            return ''

        row = [
            f'{time.monotonic() - started:.6f}',
            f'{float(g.get("x", 0.0)):.6f}',
            f'{float(g.get("y", 0.0)):.6f}',
            f'{float(g.get("vx", 0.0)):.6f}',
            f'{float(g.get("vy", 0.0)):.6f}',
            str(g.get('mode', '')),
            '1' if g.get('enabled', False) else '0',
            '1' if g.get('homed', False) else '0',
            fval(pose_e, 'pitch_deg'),
            fval(pose_e, 'roll_deg'),
            fval(pose_e, 'pitch_count'),
            fval(pose_e, 'roll_count'),
            fval(pose_e_rel, 'x_rel_m'),
            fval(pose_e_rel, 'y_rel_m'),
            fval(pose_e_rel, 'z_rel_m'),
            fval(pose_e_rel, 'vx_rel_m_s'),
            fval(pose_e_rel, 'vy_rel_m_s'),
            fval(pose_e_rel, 'vz_rel_m_s'),
        ]
        with self._csv_log_lock:
            if self._csv_log_writer is None:
                return
            self._csv_log_writer.writerow(row)
            self._csv_log_rows += 1
            if self._csv_log_rows % 50 == 0 and self._csv_log_fp is not None:
                self._csv_log_fp.flush()

    def _on_gantry_state(self, msg: GantryState) -> None:
        now = time.monotonic()
        homed = bool(msg.homed)
        with self._gantry_lock:
            self._gantry_last_rx = now
            self._gantry_rx_times.append(now)
            self._gantry_loop_rate_hz = float(msg.loop_rate_hz)
            just_homed = homed and not self._prev_homed
            self._prev_homed = homed
            self.gantry = {
                'x': float(msg.x),
                'y': float(msg.y),
                'vx': float(msg.vx),
                'vy': float(msg.vy),
                'mode': str(msg.mode),
                'enabled': bool(msg.enabled),
                'homed': homed,
                'homing_active': bool(msg.homing_active),
                'homing_status': str(msg.homing_status),
                'estop': bool(msg.estop),
                'move_done': bool(msg.move_done),
            }
        # Re-zero the vision session origin on the homed false->true edge so the
        # camera reports (0, 0) at the home position. Covers manual, sensor, and
        # MISSION auto-home paths. Published outside the lock.
        if just_homed:
            self._reset_vision_pub.publish(Empty())
            self._reset_drift_stats()
            self.get_logger().info(
                'Gantry homed — published /payload/reset_origin (camera re-zeroed)')

    def _on_pose_e(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 3:
            return
        now = time.monotonic()
        with self._gantry_lock:
            self._pose_e_last_rx = now
            self._pose_e_rx_times.append(now)
            self.pose_e = {
                'time': float(msg.data[0]),
                'pitch_deg': float(msg.data[1]),
                'roll_deg': float(msg.data[2]),
            }
            if len(msg.data) >= 5:
                self.pose_e['pitch_count'] = float(msg.data[3])
                self.pose_e['roll_count'] = float(msg.data[4])

    def _on_ic_cart_state(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 5:
            return
        with self._gantry_lock:
            self._ic_cart_last_rx = time.monotonic()
            self.ic_cart = {
                'step_pos_mm': float(msg.data[1]),
                'enc_pos_mm': float(msg.data[2]),
                'target_mm': float(msg.data[3]),
                'moving': bool(msg.data[4] >= 0.5),
                'stale': bool(msg.data[6] >= 0.5) if len(msg.data) >= 7 else False,
            }

    def _reset_drift_stats(self) -> None:
        """Restart drift statistics — called on camera calibration and homing."""
        with self._gantry_lock:
            self._drift = {
                'dx_mm': None, 'dy_mm': None,
                'ewma_dx_mm': None, 'ewma_dy_mm': None,
                'peak_dx_mm': 0.0, 'peak_dy_mm': 0.0,
                'n_samples': 0,
                'cal_t': time.monotonic(),
                'last_sample_t': 0.0,
                'sampling': False,
            }
            self._drift_history.clear()
            self._drift_last_hist_t = 0.0

    def _update_drift_locked(self, now: float) -> None:
        """Sample camera↔encoder delta.  Caller holds _gantry_lock.

        Gated on quiescence: the instantaneous delta includes real payload
        swing (camera tracks the payload, motor encoders track the cart), so
        only samples taken with the gantry idle and the rope swing settled
        measure actual sensor drift."""
        d = self._drift
        if not d:
            return
        quiescent = bool(self.gantry.get('move_done', False))
        if quiescent and self.pose_e_rel and (now - self._pose_e_rel_last_rx) < 2.0:
            quiescent = (
                abs(float(self.pose_e_rel.get('vx_rel_m_s', 0.0))) < self._drift_swing_settle_m_s
                and abs(float(self.pose_e_rel.get('vy_rel_m_s', 0.0))) < self._drift_swing_settle_m_s
            )
        gantry_fresh = (self._gantry_last_rx > 0.0
                        and (now - self._gantry_last_rx) < 2.0)
        p = self.payload
        gx = float(p.get('gantry_x', float('nan'))) if p else float('nan')
        gy = float(p.get('gantry_y', float('nan'))) if p else float('nan')
        sampling = (quiescent and gantry_fresh and bool(p and p.get('valid'))
                    and math.isfinite(gx) and math.isfinite(gy))
        d['sampling'] = sampling
        if not sampling:
            d['last_sample_t'] = 0.0
            return
        dx_mm = (float(self.gantry['x']) - gx) * 1000.0
        dy_mm = (float(self.gantry['y']) - gy) * 1000.0
        d['dx_mm'] = dx_mm
        d['dy_mm'] = dy_mm
        if d['ewma_dx_mm'] is None:
            d['ewma_dx_mm'] = dx_mm
            d['ewma_dy_mm'] = dy_mm
        else:
            dt = (now - d['last_sample_t']) if d['last_sample_t'] > 0.0 else 0.05
            a = 1.0 - math.exp(-max(dt, 1e-3) / self._drift_ewma_tau_s)
            d['ewma_dx_mm'] += a * (dx_mm - d['ewma_dx_mm'])
            d['ewma_dy_mm'] += a * (dy_mm - d['ewma_dy_mm'])
        d['last_sample_t'] = now
        d['n_samples'] += 1
        if abs(dx_mm) > abs(d['peak_dx_mm']):
            d['peak_dx_mm'] = dx_mm
        if abs(dy_mm) > abs(d['peak_dy_mm']):
            d['peak_dy_mm'] = dy_mm
        if now - self._drift_last_hist_t >= 1.0:
            self._drift_last_hist_t = now
            self._drift_history.append(
                (round(now - d['cal_t'], 1), round(dx_mm, 1), round(dy_mm, 1)))

    def _on_payload_state(self, msg: PayloadState) -> None:
        now = time.monotonic()
        with self._gantry_lock:
            self._payload_last_rx = now
            self._payload_rx_times.append(now)
            self.payload = {
                'motion_time_sec': float(msg.motion_time_sec),
                'x1': float(msg.x1),
                'z1': float(msg.z1),
                'x2': float(msg.x2),
                'z2': float(msg.z2),
                'vx1': float(msg.vx1),
                'vz1': float(msg.vz1),
                'vx2': float(msg.vx2),
                'vz2': float(msg.vz2),
                'cam_x': float(msg.cam_x),
                'cam_y': float(msg.cam_y),
                'cam_z': float(msg.cam_z),
                'gantry_x': float(msg.gantry_x),
                'gantry_y': float(msg.gantry_y),
                'gantry_z': float(msg.gantry_z),
                'frame_id': str(msg.frame_id),
                'valid': bool(msg.valid),
                'interpolated': bool(msg.interpolated),
            }
            self._update_drift_locked(now)

    def _on_pose_e_rel(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 9:
            return
        now = time.monotonic()
        with self._gantry_lock:
            self._pose_e_rel_last_rx = now
            self._pose_e_rel_rx_times.append(now)
            self.pose_e_rel = {
                'time': float(msg.data[0]),
                'pitch_deg': float(msg.data[1]),
                'roll_deg': float(msg.data[2]),
                'x_rel_m': float(msg.data[3]),
                'y_rel_m': float(msg.data[4]),
                'z_rel_m': float(msg.data[5]),
                'vx_rel_m_s': float(msg.data[6]),
                'vy_rel_m_s': float(msg.data[7]),
                'vz_rel_m_s': float(msg.data[8]),
            }

    def _on_stack_sync_hz(self, msg: Float64) -> None:
        with self._gantry_lock:
            self._stack_sync_hz = float(msg.data)
            self._stack_sync_last_rx = time.monotonic()

    def _on_camera(self, msg: CompressedImage) -> None:
        if not msg.data:
            return
        with self._media_lock:
            self._camera_last_rx = time.monotonic()
            self._camera_jpeg = bytes(msg.data)
            self._camera_frame_id = getattr(self, "_camera_frame_id", 0) + 1

    def _wait_future(self, future, timeout: float) -> bool:
        """Wait for an async service future without spin_until_future_complete.

        The main MultiThreadedExecutor thread must keep spinning; calling
        spin_until_future_complete from HTTP/mission threads can starve subscriptions.
        """
        done = threading.Event()

        def _on_done(_future) -> None:
            done.set()

        future.add_done_callback(_on_done)
        return done.wait(timeout)

    def get_state(self) -> Dict[str, Any]:
        with self._gantry_lock:
            g = dict(self.gantry)
            pose_e = dict(self.pose_e) if self.pose_e else None
            pose_e_rel = dict(self.pose_e_rel) if self.pose_e_rel else None
            payload = dict(self.payload) if self.payload else None
            wps = list(self.waypoints)
            running = self.mission_running
            gantry_last_rx = self._gantry_last_rx
            payload_last_rx = self._payload_last_rx
            pose_e_last_rx = self._pose_e_last_rx
            pose_e_rel_last_rx = self._pose_e_rel_last_rx
            gantry_rx_times = self._gantry_rx_times
            payload_rx_times = self._payload_rx_times
            pose_e_rx_times = self._pose_e_rx_times
            pose_e_rel_rx_times = self._pose_e_rel_rx_times
            gantry_loop_rate_hz = self._gantry_loop_rate_hz
            stack_sync_hz = self._stack_sync_hz
            stack_sync_last_rx = self._stack_sync_last_rx
            drift = dict(self._drift) if self._drift else None
            drift_history = list(self._drift_history)
            ic_cart = dict(self.ic_cart) if self.ic_cart else None
            ic_cart_last_rx = self._ic_cart_last_rx
        with self._media_lock:
            camera_last_rx = self._camera_last_rx
            camera_frame_id = int(getattr(self, '_camera_frame_id', 0))
        now = time.monotonic()
        stale_s = self._gantry_stale_s
        gantry_ever = gantry_last_rx > 0.0
        gantry_live = gantry_ever and (now - gantry_last_rx) < stale_s
        gantry_stale_s = (
            round(now - gantry_last_rx, 2) if gantry_ever else None
        )
        payload_live = payload_last_rx > 0.0 and (now - payload_last_rx) < 2.0
        pose_e_live = pose_e_last_rx > 0.0 and (now - pose_e_last_rx) < 2.0
        pose_e_rel_live = (
            pose_e_rel_last_rx > 0.0 and (now - pose_e_rel_last_rx) < 2.0
        )
        camera_live = camera_last_rx > 0.0 and (now - camera_last_rx) < 2.0

        win = self._rate_window_s
        gantry_hz = self._estimate_hz(gantry_rx_times, now, win)
        payload_hz = self._estimate_hz(payload_rx_times, now, win)
        pose_e_hz = self._estimate_hz(pose_e_rx_times, now, win)
        pose_e_rel_hz = self._estimate_hz(pose_e_rel_rx_times, now, win)
        target_hz = float(self.stack_pose_publish_hz)
        sync_live = stack_sync_last_rx > 0.0 and (now - stack_sync_last_rx) < 3.0
        ref_hz = stack_sync_hz if sync_live and stack_sync_hz > 0.0 else target_hz
        # Use measured RX times for all streams (loop_rate_hz is publish EMA, not RX).
        tol = max(3.0, 0.10 * ref_hz)
        pose_rates_synced = (
            ref_hz > 0.0
            and gantry_hz > 0.0
            and payload_hz > 0.0
            and abs(gantry_hz - ref_hz) <= tol
            and abs(payload_hz - ref_hz) <= tol
            and abs(gantry_hz - payload_hz) <= tol
        )
        if pose_e_live and pose_e_hz > 0.0:
            pose_rates_synced = (
                pose_rates_synced and abs(pose_e_hz - ref_hz) <= tol
            )

        camera_drift = None
        if drift is not None:
            def _r1(v):
                return round(v, 1) if isinstance(v, float) else v
            camera_drift = {
                'dx_mm': _r1(drift['dx_mm']),
                'dy_mm': _r1(drift['dy_mm']),
                'ewma_dx_mm': _r1(drift['ewma_dx_mm']),
                'ewma_dy_mm': _r1(drift['ewma_dy_mm']),
                'peak_dx_mm': _r1(drift['peak_dx_mm']),
                'peak_dy_mm': _r1(drift['peak_dy_mm']),
                'n_samples': int(drift['n_samples']),
                # A frozen camera stream can't be sampling, whatever the last
                # payload callback decided.
                'sampling': bool(drift['sampling']) and payload_live,
                't_since_cal_s': round(now - float(drift['cal_t']), 1),
                'history': drift_history,
            }

        px, py = float(g['x']), float(g['y'])
        position_source = 'gantry'
        if not gantry_live and payload:
            gx = float(payload.get('gantry_x', float('nan')))
            gy = float(payload.get('gantry_y', float('nan')))
            if math.isfinite(gx) and math.isfinite(gy):
                px, py = gx, gy
                position_source = 'payload'

        return {
            'position': {'x': px, 'y': py},
            'position_source': position_source,
            'velocity': {'vx': g['vx'], 'vy': g['vy']},
            'mode': g['mode'],
            'enabled': g['enabled'],
            'homed': g['homed'],
            'homing_active': g.get('homing_active', False),
            'homing_status': g.get('homing_status', ''),
            'estop': g['estop'],
            'move_done': g['move_done'],
            'mission_running': running,
            'workspace_m': self.workspace_m,
            'csv_dir': self.csv_dir,
            'csv_profiles': self._list_csv_profiles(),
            'csv_log': self._csv_log_state(),
            'pose_e': pose_e,
            'pose_e_rel': pose_e_rel,
            'payload': payload,
            'waypoints': wps,
            'camera_available': True,
            'gantry_live': gantry_live,
            'gantry_ever_received': gantry_ever,
            'gantry_stale_s': gantry_stale_s,
            'payload_live': payload_live,
            'payload_stale_s': (
                round(now - payload_last_rx, 2) if payload_last_rx > 0.0 else None
            ),
            'camera_drift': camera_drift,
            'pose_e_live': pose_e_live,
            'pose_e_hz': round(pose_e_hz, 1),
            'pose_e_rel_live': pose_e_rel_live,
            'pose_e_rel_hz': round(pose_e_rel_hz, 1),
            'camera_live': camera_live,
            'camera_frame_id': camera_frame_id,
            'ic_cart': ic_cart,
            'ic_cart_live': (
                ic_cart_last_rx > 0.0 and (now - ic_cart_last_rx) < 2.0
            ),
            'goto_status': dict(self._goto_status),
            'stack_pose_publish_hz': target_hz,
            'stack_sync_hz': round(stack_sync_hz, 1) if sync_live else None,
            'stack_sync_live': sync_live,
            'gantry_hz': round(gantry_hz, 1),
            'payload_hz': round(payload_hz, 1),
            'pose_rates_synced': pose_rates_synced,
            'experiment': self._experiment_state,
        }

    def _payload_fresh(self, max_age_s: float = 2.0) -> bool:
        """True if /payload/state (camera tracking) arrived recently."""
        with self._gantry_lock:
            last_rx = self._payload_last_rx
        return last_rx > 0.0 and (time.monotonic() - last_rx) < max_age_s

    def _call_trigger(self, client, timeout: float = 5.0) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f'Service {client.srv_name} not available'
        future = client.call_async(Trigger.Request())
        if not self._wait_future(future, timeout):
            return False, 'Service call timeout'
        if not future.done() or future.result() is None:
            return False, 'Service call failed'
        r = future.result()
        return bool(r.success), str(r.message)

    def _call_ic_cart_move(
        self,
        position_mm: float,
        velocity_mm_s: float,
        timeout: float = 5.0,
    ) -> tuple[bool, str]:
        client = self._ic_cart_move_cli
        if not client.wait_for_service(timeout_sec=2.0):
            return False, 'IC cart node not running'
        req = IcCartMove.Request()
        req.position_mm = float(position_mm)
        req.velocity_mm_s = float(velocity_mm_s)
        future = client.call_async(req)
        if not self._wait_future(future, timeout):
            return False, 'IC cart move timeout'
        if not future.done() or future.result() is None:
            return False, 'IC cart move failed'
        r = future.result()
        return bool(r.success), str(r.message)

    def _set_encoder_float_param(
        self,
        name: str,
        value: float,
        timeout: float = 5.0,
    ) -> tuple[bool, str]:
        param = Parameter()
        param.name = name
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_DOUBLE
        param.value.double_value = float(value)

        for client in (self._encoder_serial_params_cli, self._encoder_gpio_params_cli):
            if not client.wait_for_service(timeout_sec=0.2):
                continue
            req = SetParameters.Request()
            req.parameters = [param]
            future = client.call_async(req)
            if not self._wait_future(future, timeout):
                return False, f'{client.srv_name} timeout'
            if not future.done() or future.result() is None:
                return False, f'{client.srv_name} failed'
            results = future.result().results
            if not results:
                return False, f'{client.srv_name} returned no result'
            result = results[0]
            if result.successful:
                return True, f'{name} set to {value:.3f}'
            return False, result.reason or f'{client.srv_name} rejected {name}'
        return False, 'No encoder parameter service available'

    def _get_gantry_move_vel(self, timeout: float = 3.0) -> Optional[float]:
        """Read the controller's current mission_move_vel_ms (m/s), or None."""
        client = self._gantry_get_params_cli
        if not client.wait_for_service(timeout_sec=1.0):
            return None
        req = GetParameters.Request()
        req.names = ['mission_move_vel_ms']
        future = client.call_async(req)
        if not self._wait_future(future, timeout):
            return None
        if not future.done() or future.result() is None or not future.result().values:
            return None
        val = future.result().values[0]
        if val.type != ParameterType.PARAMETER_DOUBLE:
            return None
        return float(val.double_value)

    def _set_gantry_move_vel(self, vel_ms: float,
                             timeout: float = 3.0) -> tuple[bool, str]:
        """Set the controller's mission_move_vel_ms at runtime (clamped there)."""
        param = Parameter()
        param.name = 'mission_move_vel_ms'
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_DOUBLE
        param.value.double_value = float(vel_ms)
        client = self._gantry_params_cli
        if not client.wait_for_service(timeout_sec=1.0):
            return False, 'Gantry parameter service not available'
        req = SetParameters.Request()
        req.parameters = [param]
        future = client.call_async(req)
        if not self._wait_future(future, timeout):
            return False, 'set_parameters timeout'
        if not future.done() or future.result() is None or not future.result().results:
            return False, 'set_parameters failed'
        r = future.result().results[0]
        if r.successful:
            return True, f'move vel set to {vel_ms:.3f} m/s'
        return False, r.reason or 'set_parameters rejected'

    def _start_experiment_run_async(self, axis: str, direction: float,
                                    displacement_mm: float,
                                    velocity_mm_s: float) -> None:
        """Commanded-move experiment: set speed, record encoders+camera over a
        single move of ±displacement along one axis, then hold the data for a
        titled save (experiment_save action → /experiment/save)."""
        def run() -> None:
            prev_vel: Optional[float] = None
            recording = False
            try:
                with self._motion_lock:
                    if self._goto_running:
                        self._set_goto_status('error', 'Motion already running')
                        return
                    self._goto_running = True
                self._experiment_run_active = True
                with self._gantry_lock:
                    gx = float(self.gantry.get('x', 0.0))
                    gy = float(self.gantry.get('y', 0.0))
                disp_m = direction * displacement_mm / 1000.0
                tx, ty = gx, gy
                if axis == 'x':
                    tx = min(max(gx + disp_m, 0.0), self.workspace_m)
                else:
                    ty = min(max(gy + disp_m, 0.0), self.workspace_m)
                actual_mm = ((tx - gx) if axis == 'x' else (ty - gy)) * 1000.0
                if abs(actual_mm) < 1.0:
                    self._set_goto_status(
                        'error', 'Experiment move clamps to zero — at workspace edge')
                    return
                prev_vel = self._get_gantry_move_vel()
                ok, msg = self._set_gantry_move_vel(velocity_mm_s / 1000.0)
                if not ok:
                    self._set_goto_status('error', f'Velocity set failed: {msg}')
                    return
                self._set_goto_status(
                    'running',
                    f'Experiment: {actual_mm:+.0f} mm {axis.upper()} '
                    f'@ {velocity_mm_s:.0f} mm/s')
                self._exp_start_pub.publish(Empty())
                self._experiment_state = 'recording'
                recording = True
                time.sleep(0.5)          # pre-move baseline in the recording
                ok, msg = self._call_move_to(tx, ty)
                if not ok:
                    self._set_goto_status('error', f'Experiment move failed: {msg}')
                    return
                ok_move, wait_msg = self._wait_move_done(tx, ty)
                time.sleep(10.0)         # capture post-move residual swing
                self._exp_stop_pub.publish(Empty())
                recording = False
                self._experiment_state = 'awaiting_save'
                if ok_move:
                    self._set_goto_status(
                        'done', 'Experiment run complete — title it and Save')
                else:
                    self._set_goto_status(
                        'error',
                        f'{wait_msg or "Move timed out"} — data held, you can still Save')
            finally:
                if recording:
                    # Bail-out path: stop recording so the tracker isn't left
                    # logging forever; data stays held (save or overwrite).
                    self._exp_stop_pub.publish(Empty())
                    self._experiment_state = 'idle'
                if prev_vel is not None:
                    self._set_gantry_move_vel(prev_vel)
                self._experiment_run_active = False
                with self._motion_lock:
                    self._goto_running = False

        threading.Thread(target=run, daemon=True, name='experiment_run').start()

    def _call_set_mode(
        self,
        mode: str,
        *,
        csv_path: str = '',
        jog_preset: int = 0,
        target_x: float = 0.0,
        target_y: float = 0.0,
        timeout: float = 30.0,
    ) -> tuple[bool, str]:
        if not self._set_mode_cli.wait_for_service(timeout_sec=2.0):
            return False, 'Service /gantry/set_mode not available'
        req = SetMode.Request()
        req.mode = mode
        req.csv_path = csv_path
        req.jog_speed_preset = int(jog_preset)
        req.target_x = float(target_x)
        req.target_y = float(target_y)
        future = self._set_mode_cli.call_async(req)
        if not self._wait_future(future, timeout):
            return False, 'set_mode call timeout'
        if not future.done() or future.result() is None:
            return False, 'set_mode call failed'
        r = future.result()
        return bool(r.success), str(r.message)

    def _call_move_to(self, x: float, y: float, timeout: float = 5.0) -> tuple[bool, str]:
        if not self._move_to_cli.wait_for_service(timeout_sec=2.0):
            return False, 'Service /gantry/move_to not available'
        req = MoveTo.Request()
        req.x = float(x)
        req.y = float(y)
        future = self._move_to_cli.call_async(req)
        if not self._wait_future(future, timeout):
            return False, 'move_to call timeout'
        if not future.done() or future.result() is None:
            return False, 'move_to call failed'
        r = future.result()
        return bool(r.success), str(r.message)

    def _prepare_for_motion(self) -> tuple[bool, str]:
        """Enable motors, enter MISSION mode (re-home only if not yet homed)."""
        with self._gantry_lock:
            if self.gantry.get('estop', False):
                return False, 'E-stop active — clear E-stop first'
            enabled = bool(self.gantry.get('enabled', False))
            homed = bool(self.gantry.get('homed', False))
            homing_active = bool(self.gantry.get('homing_active', False))
        if homing_active:
            return False, 'Homing already in progress'
        if not enabled:
            ok, msg = self._call_trigger(self._enable_cli)
            if not ok:
                return False, f'Enable failed: {msg}'
        if not homed:
            return False, 'Not homed — run Home first'
        ok, msg = self._call_set_mode('MISSION', timeout=120.0)
        if not ok:
            return False, msg
        with self._gantry_lock:
            need_wait = bool(self.gantry.get('homing_active', False))
            homed = bool(self.gantry.get('homed', False))
            mode = str(self.gantry.get('mode', ''))
        if msg and 'homing' in msg.lower():
            need_wait = True
        if need_wait or mode == 'HOMING':
            if not self._wait_sensor_homing():
                return False, 'Sensor homing failed or timed out'
            with self._gantry_lock:
                homed = bool(self.gantry.get('homed', False))
        elif not homed:
            return False, 'Not homed — run Home first'
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._gantry_lock:
                if str(self.gantry.get('mode', '')) == 'MISSION':
                    return True, 'Ready for motion'
            time.sleep(0.05)
        return False, 'Gantry did not enter MISSION mode — retry Go to'

    def handle_action(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if name == 'enable':
            ok, msg = self._call_trigger(self._enable_cli)
        elif name == 'disable':
            ok, msg = self._call_trigger(self._disable_cli)
        elif name == 'home':
            ok, msg = self._run_sensor_homing()
        elif name == 'estop':
            ok, msg = self._call_trigger(self._estop_cli)
        elif name == 'clear_estop':
            ok, msg = self._call_trigger(self._clear_estop_cli)
        elif name == 'traj_arm':
            ok, msg = self._call_set_mode('TRAJ')
        elif name == 'stabilize_payload':
            ok, msg = self._start_payload_stabilizer()
        elif name == 'start_csv_log':
            ok, msg = self._start_csv_log()
        elif name == 'stop_csv_log':
            ok, msg = self._stop_csv_log()
        elif name == 'set_mode':
            mode = str(data.get('mode', 'IDLE')).upper()
            jog_preset = int(data.get('jog_preset', 0))
            csv_path = str(data.get('csv_path', ''))
            ok, msg = self._call_set_mode(mode, csv_path=csv_path, jog_preset=jog_preset)
        elif name in ('move_to', 'goto'):
            x = float(data.get('x', 0.0))
            y = float(data.get('y', 0.0))
            if not (0.0 <= x <= self.workspace_m and 0.0 <= y <= self.workspace_m):
                return {
                    'status': 'error',
                    'message': (
                        f'Target ({x:.3f}, {y:.3f}) outside workspace '
                        f'[0, {self.workspace_m:.2f}] m — click inside the map bounds'
                    ),
                }
            if name == 'goto':
                self._start_goto_async(x, y)
                return {
                    'status': 'ok',
                    'message': f'Moving to ({x:.3f}, {y:.3f}) m — watch cart on map',
                }
            ok, msg = self._prepare_for_motion()
            if not ok:
                return {'status': 'error', 'message': msg}
            ok, msg = self._call_move_to(x, y)
            if not ok:
                return {'status': 'error', 'message': msg}
        elif name == 'reset_vision':
            self._reset_vision_pub.publish(Empty())
            ok, msg = True, 'Vision session origin reset'
        elif name == 'calibrate_payload':
            self._reset_vision_pub.publish(Empty())
            enc_ok, enc_msg = self._call_trigger(self._encoder_reset_cli)
            if enc_ok:
                ok = True
                msg = 'Payload reference calibrated: camera + encoder origins reset'
            else:
                ok = False
                msg = f'Camera origin reset; encoder reset failed: {enc_msg}'
        elif name == 'set_rope_length':
            rope_length_m = float(data.get('rope_length_m', 0.0))
            if not (0.05 <= rope_length_m <= 5.0):
                return {
                    'status': 'error',
                    'message': 'Rope length must be between 0.05 and 5.0 m',
                }
            ok, msg = self._set_encoder_float_param(
                'rope_length_m', rope_length_m)
        elif name == 'experiment_start':
            self._exp_start_pub.publish(Empty())
            self._experiment_state = 'recording'
            ok, msg = True, 'Experiment started'
        elif name == 'experiment_end':
            self._exp_end_pub.publish(Empty())
            self._experiment_state = 'saved'
            ok, msg = True, 'Experiment ended — saving data'
        elif name == 'experiment_run':
            try:
                axis = str(data.get('axis', 'x')).lower()
                direction = 1.0 if float(data.get('direction', 1)) >= 0 else -1.0
                disp = float(data.get('displacement_mm', 0.0))
                vel = float(data.get('velocity_mm_s', 0.0))
            except (TypeError, ValueError):
                return {'status': 'error', 'message': 'Bad experiment parameters'}
            if axis not in ('x', 'y'):
                return {'status': 'error', 'message': "axis must be 'x' or 'y'"}
            if not (1.0 <= disp <= 5000.0):
                return {'status': 'error',
                        'message': 'Displacement must be 1–5000 mm'}
            if not (5.0 <= vel <= 1000.0):
                return {'status': 'error',
                        'message': 'Velocity must be 5–1000 mm/s'}
            with self._gantry_lock:
                homed = bool(self.gantry.get('homed', False))
                enabled = bool(self.gantry.get('enabled', False))
            if not (homed and enabled):
                return {'status': 'error',
                        'message': 'Gantry must be homed and enabled first'}
            if self._goto_running or self.mission_running:
                return {'status': 'error', 'message': 'Motion already running'}
            self._start_experiment_run_async(axis, direction, disp, vel)
            ok = True
            msg = (f'Experiment run started: {direction * disp:+.0f} mm '
                   f'{axis.upper()} @ {vel:.0f} mm/s')
        elif name == 'experiment_save':
            title = str(data.get('title', '')).strip()
            m = String()
            m.data = title
            self._exp_save_pub.publish(m)
            self._experiment_state = 'saved'
            ok = True
            msg = f'Experiment saved — {title}' if title else 'Experiment saved'
        elif name == 'set_start_x':
            val = float(data.get('start_x_mm', 350.0))
            f64 = Float64()
            f64.data = val
            self._exp_start_x_pub.publish(f64)
            ok, msg = True, f'Start X offset set to {val:.1f} mm'
        elif name == 'cal_origin':
            self._reset_vision_pub.publish(Empty())
            self._reset_drift_stats()
            enc_ok, _ = self._call_trigger(self._encoder_reset_cli)
            with self._gantry_lock:
                gx = float(self.gantry.get('x', 0.0))
                gy = float(self.gantry.get('y', 0.0))
            ok = True
            msg = f'Origin set @ ({gx * 1000:.0f}, {gy * 1000:.0f}) mm'
            if not self._payload_fresh():
                msg += ' — camera not tracking, applies on next detection'
            if not enc_ok:
                msg += ' (encoder reset skipped)'
        elif name == 'cal_camera_ref':
            self._reset_vision_pub.publish(Empty())
            self._reset_drift_stats()
            with self._gantry_lock:
                gx = float(self.gantry.get('x', 0.0))
                gy = float(self.gantry.get('y', 0.0))
            ok = True
            if self._payload_fresh():
                msg = f'Camera origin set @ ({gx * 1000:.0f}, {gy * 1000:.0f}) mm'
            else:
                msg = ('Calibration queued — camera not tracking '
                       '(applies on next detection)')
        elif name == 'cal_encoder_ref':
            ok, msg = self._call_trigger(self._encoder_reset_cli)
            if ok:
                msg = 'Encoder origin reset'
        elif name == 'ic_cart_calibrate':
            ok, msg = self._call_trigger(self._ic_cart_cal_cli)
        elif name == 'ic_cart_move':
            ok, msg = self._call_ic_cart_move(
                float(data.get('position_mm', 0.0)),
                float(data.get('velocity_mm_s', 0.0)),
            )
        elif name == 'ic_cart_stop':
            ok, msg = self._call_trigger(self._ic_cart_stop_cli)
        elif name == 'ic_cart_enable':
            ok, msg = self._call_trigger(self._ic_cart_enable_cli)
        elif name == 'ic_cart_disable':
            ok, msg = self._call_trigger(self._ic_cart_disable_cli)
        else:
            return {'status': 'error', 'message': f'Unknown action: {name}'}
        return {'status': 'ok' if ok else 'error', 'message': msg}

    def set_waypoints(self, wps: List[Dict[str, Any]]) -> None:
        with self._gantry_lock:
            self.waypoints = list(wps)

    def clear_waypoints(self) -> None:
        with self._gantry_lock:
            self.waypoints = []

    def stop_mission(self) -> None:
        self._mission_stop.set()
        self.mission_running = False
        self._call_set_mode('IDLE')

    def safe_shutdown_gantry(self) -> None:
        """Best-effort motor disable when dashboard exits (Ctrl+C on launch)."""
        self._stop_csv_log()
        self._stop_payload_stabilizer()
        self.get_logger().warn(
            'Dashboard shutdown — disabling gantry motors (/gantry/disable)')
        try:
            self._call_set_mode('IDLE', timeout=3.0)
        except Exception:
            pass
        ok, msg = self._call_trigger(self._disable_cli, timeout=5.0)
        if ok:
            self.get_logger().info(f'Gantry disabled: {msg}')
        else:
            self.get_logger().warn(f'Gantry disable on exit: {msg}')

    def _wait_sensor_homing(self) -> bool:
        """Wait until gantry sensor homing finishes (auto-home before MISSION)."""
        deadline = time.monotonic() + self.homing_timeout_s
        while time.monotonic() < deadline:
            if self._mission_stop.is_set():
                return False
            with self._gantry_lock:
                active = bool(self.gantry.get('homing_active', False))
                homed = bool(self.gantry.get('homed', False))
                mode = str(self.gantry.get('mode', ''))
            if not active and homed and mode == 'MISSION':
                return True
            if not active and homed and mode == 'IDLE':
                # Manual home finished while waiting for MISSION auto-home — keep waiting
                pass
            elif not active and mode == 'IDLE' and not homed:
                self.get_logger().error('Homing failed — gantry returned IDLE without homed')
                return False
            elif not active and mode == 'HOMING' and homed:
                # Transient: finalize may set homed before mode leaves HOMING
                pass
            time.sleep(0.05)
        self.get_logger().error('Sensor homing timeout')
        return False

    def _wait_move_done(
        self,
        target_x: Optional[float] = None,
        target_y: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Wait for move_done; handle fast completion and already-at-target."""
        time.sleep(0.1)
        tol = self.position_tolerance_m
        deadline = time.monotonic() + self.move_timeout_s
        never_started_at = time.monotonic() + 1.0
        saw_busy = False
        while time.monotonic() < deadline:
            if self._mission_stop.is_set():
                return False, 'Stopped'
            with self._gantry_lock:
                done = bool(self.gantry.get('move_done', True))
                gx = float(self.gantry.get('x', 0.0))
                gy = float(self.gantry.get('y', 0.0))
            at_target = (
                target_x is not None
                and target_y is not None
                and abs(gx - target_x) <= tol
                and abs(gy - target_y) <= tol
            )
            if not done:
                saw_busy = True
            elif saw_busy or at_target:
                return True, ''
            if (
                not saw_busy
                and not at_target
                and time.monotonic() > never_started_at
            ):
                return False, 'Move never started — check MISSION mode and motors'
            time.sleep(0.05)
        self.get_logger().error('Move timeout')
        return False, 'Move timed out'

    def _run_sensor_homing(self) -> tuple[bool, str]:
        """Start sensor homing (non-blocking); UI polls /state for completion."""
        with self._gantry_lock:
            if self.gantry.get('estop', False):
                return False, 'E-stop active — Clear E-stop, then Home'
            if self.gantry.get('homing_active', False):
                return True, 'Homing already in progress'
            enabled = bool(self.gantry.get('enabled', False))
        if not enabled:
            ok, msg = self._call_trigger(self._enable_cli)
            if not ok:
                return False, f'Enable failed: {msg}'
        ok, msg = self._call_set_mode('HOME', timeout=30.0)
        if not ok:
            return False, msg
        return True, msg or 'Sensor homing started (back → left)'

    def _start_payload_stabilizer(self) -> tuple[bool, str]:
        if self._stabilizer_proc is not None and self._stabilizer_proc.poll() is None:
            return False, 'Payload stabilizer already running'
        with self._gantry_lock:
            if self._goto_running or self.mission_running:
                return False, 'Wait for the current move to finish before stabilizing payload'
            if self.gantry.get('estop', False):
                return False, 'E-stop active — clear E-stop first'
            if not self.gantry.get('homed', False):
                return False, 'Not homed — run Home first'
            if not self.gantry.get('move_done', True):
                return False, 'Gantry is still moving — wait for arrival first'

        log_dir = Path('/home/sanjay/crane_ws/log/payload_stabilizer')
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = log_dir / f'dashboard_both_{stamp}.csv'
        terminal_log_path = log_dir / f'dashboard_both_{stamp}.log'
        command_text = (
            'cd /home/sanjay/crane_ws\n'
            'source install/setup.bash\n'
            'ros2 run gantry_control payload_stabilizer.py '
            f'--log-csv {csv_path}'
        )
        cmd = ['bash', '-lc', command_text]
        try:
            terminal_log = terminal_log_path.open('w')
            terminal_log.write('$ ' + command_text.replace('\n', '\n$ ') + '\n\n')
            terminal_log.flush()
            self._stabilizer_proc = subprocess.Popen(
                cmd,
                cwd='/home/sanjay/crane_ws',
                start_new_session=True,
                stdout=terminal_log,
                stderr=subprocess.STDOUT,
            )
            terminal_log.close()
        except Exception as exc:
            self._stabilizer_proc = None
            return False, f'Failed to start payload stabilizer: {exc}'
        self.get_logger().info(
            f'Payload stabilizer started pid={self._stabilizer_proc.pid} log={csv_path}')
        return True, f'Payload stabilizer started ({csv_path})'

    def _stop_payload_stabilizer(self) -> None:
        proc = self._stabilizer_proc
        if proc is None or proc.poll() is not None:
            self._stabilizer_proc = None
            return
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        self._stabilizer_proc = None

    def _set_goto_status(self, state: str, message: str) -> None:
        with self._gantry_lock:
            self._goto_status = {'state': state, 'message': message}

    def _start_goto_async(self, x: float, y: float) -> None:
        def run() -> None:
            with self._motion_lock:
                if self._goto_running:
                    self._set_goto_status('error', 'Go-to already running')
                    return
                self._goto_running = True
            self._set_goto_status('running', f'Moving to ({x:.3f}, {y:.3f}) m')
            try:
                ok, msg = self._prepare_for_motion()
                if not ok:
                    self.get_logger().error(f'Go-to prepare failed: {msg}')
                    self._set_goto_status('error', msg)
                    return
                ok, msg = self._call_move_to(x, y)
                if not ok:
                    self.get_logger().error(f'Go-to move failed: {msg}')
                    self._set_goto_status('error', msg)
                    return
                ok_move, wait_msg = self._wait_move_done(x, y)
                if ok_move:
                    done_msg = f'At ({x:.3f}, {y:.3f}) m'
                    self.get_logger().info(f'Go-to complete {done_msg}')
                    self._set_goto_status('done', done_msg)
                else:
                    self.get_logger().error(f'Go-to wait failed: {wait_msg}')
                    self._set_goto_status('error', wait_msg or 'Move timed out')
            finally:
                with self._motion_lock:
                    self._goto_running = False

        threading.Thread(target=run, daemon=True, name='goto').start()

    def execute_mission(self) -> None:
        with self._gantry_lock:
            wps = list(self.waypoints)
        if not wps:
            self.get_logger().warn('No waypoints')
            return

        self._mission_stop.clear()
        self.mission_running = True
        self.get_logger().info(f'Mission start: {len(wps)} waypoints')

        ok, msg = self._prepare_for_motion()
        if not ok:
            self.get_logger().error(f'Cannot start mission: {msg}')
            self.mission_running = False
            return

        for i, wp in enumerate(wps):
            if self._mission_stop.is_set():
                self.get_logger().warn('Mission stopped')
                break
            tx = float(wp.get('x', 0.0))
            ty = float(wp.get('y', 0.0))
            self.get_logger().info(f'WP {i + 1}/{len(wps)} → ({tx:.3f}, {ty:.3f}) m')
            ok, msg = self._call_move_to(tx, ty)
            if not ok:
                self.get_logger().error(msg)
                break
            ok_move, wait_msg = self._wait_move_done(tx, ty)
            if not ok_move:
                self.get_logger().error(wait_msg)
                break
            dwell = float(wp.get('dwell', 0.0))
            if dwell > 0 and not self._mission_stop.is_set():
                time.sleep(dwell)

        self.mission_running = False
        self.get_logger().info('Mission finished')

    def start_mission_async(self) -> bool:
        if self.mission_running:
            return False
        t = threading.Thread(target=self.execute_mission, daemon=True)
        t.start()
        return True


dashboard_node: Optional[CraneDashboardServer] = None


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = 'CraneDashboard/1.0'

    def log_message(self, fmt: str, *args) -> None:
        if args:
            req = str(args[0]) if args else ''
            if '/camera/' in req:
                return
            if len(args) > 1 and str(args[1]) in ('502', '404'):
                return
        if args and str(args[1]) not in ('200', '304'):
            super().log_message(fmt, *args)

    def _cors(self) -> None:
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split('?', 1)[0]
        if path in ('/', '/crane_ops.html'):
            self._serve_file('crane_ops.html', 'text/html; charset=utf-8')
        elif path == '/state':
            self._json_response(dashboard_node.get_state())
        elif path.startswith('/camera/jpeg'):
            self._serve_camera_jpeg()
        elif path.startswith('/camera/stream'):
            self._proxy_camera_stream()
        else:
            rel = path.lstrip('/')
            fpath = os.path.join(dashboard_node.web_dir, rel)
            if os.path.isfile(fpath):
                ctype = 'application/octet-stream'
                if rel.endswith('.css'):
                    ctype = 'text/css'
                elif rel.endswith('.js'):
                    ctype = 'application/javascript'
                self._serve_path(fpath, ctype)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = self.path.split('?', 1)[0]
        data = self._read_json()
        node = dashboard_node

        if path.startswith('/action/'):
            name = path[len('/action/'):].strip('/')
            self._json_response(node.handle_action(name, data))
            return
        if path == '/set_waypoints':
            node.set_waypoints(data.get('waypoints', []))
            self._json_response({'status': 'updated', 'count': len(node.waypoints)})
            return
        if path == '/execute':
            if node.mission_running:
                self._json_response({'status': 'error', 'message': 'Already running'})
                return
            if 'waypoints' in data:
                node.set_waypoints(data.get('waypoints', []))
            started = node.start_mission_async()
            self._json_response({'status': 'started' if started else 'error'})
            return
        if path == '/stop':
            node.stop_mission()
            self._json_response({'status': 'stopped'})
            return
        if path == '/clear':
            node.clear_waypoints()
            self._json_response({'status': 'cleared'})
            return

        self._json_response({'status': 'error', 'message': 'Unknown path'})

    def _read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get('Content-Length', 0))
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode('utf-8'))
        except json.JSONDecodeError:
            return {}

    def _json_response(self, obj: Dict[str, Any], code: int = 200) -> None:
        body = json.dumps(
            dashboard_node._sanitize_for_json(obj),
            allow_nan=False,
        ).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self._cors()
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, name: str, content_type: str) -> None:
        self._serve_path(os.path.join(dashboard_node.web_dir, name), content_type)

    def _serve_path(self, fpath: str, content_type: str) -> None:
        try:
            with open(fpath, 'rb') as f:
                body = f.read()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self._cors()
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_camera_jpeg(self) -> None:
        with dashboard_node._media_lock:
            jpeg = dashboard_node._camera_jpeg
            frame_id = int(getattr(dashboard_node, '_camera_frame_id', 0))
        if not jpeg:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors()
            self.end_headers()
            return
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Frame-Id', str(frame_id))
        self._cors()
        self.send_header('Content-Length', str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _proxy_camera_stream(self) -> None:
        if dashboard_node.camera_proxy:
            upstream = f'http://127.0.0.1:{dashboard_node.stream_port}/stream'
            try:
                req = urllib.request.Request(
                    upstream, headers={'Connection': 'close'})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    ctype = resp.headers.get(
                        'Content-Type',
                        'multipart/x-mixed-replace; boundary=frame')
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', ctype)
                    self.send_header(
                        'Cache-Control',
                        'no-store, no-cache, must-revalidate, max-age=0')
                    self._cors()
                    self.end_headers()
                    while True:
                        chunk = resp.read(16384)
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                return
            except (urllib.error.URLError, TimeoutError):
                # Fall through to the ROS compressed-image stream if the
                # tracker HTTP stream is not up yet.
                pass

        boundary = b"frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self._cors()
        self.end_headers()

        last_id = -1
        try:
            while True:
                with dashboard_node._media_lock:
                    jpeg = dashboard_node._camera_jpeg
                    frame_id = getattr(dashboard_node, "_camera_frame_id", 0)

                if jpeg and frame_id != last_id:
                    last_id = frame_id
                    self.wfile.write(b"--" + boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            return


def run_http_server(port: int) -> None:
    httpd = ThreadingHTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f'\n  Crane ops UI: http://0.0.0.0:{port}/\n')
    httpd.serve_forever()


def main(args=None) -> None:
    global dashboard_node
    rclpy.init(args=args)
    dashboard_node = CraneDashboardServer()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(dashboard_node)

    port = dashboard_node.dashboard_port
    http_thread = threading.Thread(target=run_http_server, args=(port,), daemon=True)
    http_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        print('\n[crane_dashboard] Ctrl+C — shutting down…')
    finally:
        if dashboard_node is not None:
            dashboard_node.safe_shutdown_gantry()
            dashboard_node.destroy_node()
        print('[crane_dashboard] Exit complete.')
        rclpy.shutdown()


if __name__ == '__main__':
    main()
