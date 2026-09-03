#!/usr/bin/env python3
"""
payload_tracker.py — DECOUPLED DETECTION + HD DISPLAY
------------------------------------------------------
- Detection: low-res mono @ max FPS (AprilTag pair on active face)
- Display: HD mono overlay (tag outlines + midpoint between pair)
- Top-down: smoothed X/Z from active face pair
- Per-tag CSV: time, tag_id, session X, dX/dt for every visible tag
- ROS2 (optional --ros): /payload/state (PayloadState: pose + velocity) + /payload/pose legacy array

Web stream at http://JETSON_IP:8080

Usage:
    python3 payload_tracker.py --ip 192.168.0.153 --marker-size 0.10
    python3 payload_tracker.py --ip 192.168.0.153 --marker-size 0.10 --display-fps 30
    python3 payload_tracker.py --ip 192.168.0.153 --ros
"""

import argparse
import csv
import json
import math
import os
import queue
import sys
import socket
import time
import threading
from collections import deque
from datetime import datetime
from typing import Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import depthai as dai
from pupil_apriltags import Detector


class _SuppressStderr:
    def __enter__(self):
        import os, sys
        sys.stderr.flush()
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._saved = os.dup(2)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, *exc):
        import os
        os.dup2(self._saved, 2)
        os.close(self._devnull)
        os.close(self._saved)


# ═══════════════════════════════════════════════════════════════════════════════
#  MJPEG WEB STREAM
# ═══════════════════════════════════════════════════════════════════════════════

class MJPEGStream:
    def __init__(self):
        self._frame = None
        self._seq = 0
        self._cond = threading.Condition()

    def update(self, frame, quality=85):
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        with self._cond:
            self._frame = jpeg.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def get(self):
        with self._cond:
            return self._frame

    def wait_next(self, last_seq=0, timeout=1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._seq, self._frame


stream = MJPEGStream()


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''<html><head>
                <title>Payload Tracker</title>
                <style>
                    body { background:#111; margin:0; display:flex;
                           justify-content:center; align-items:center;
                           height:100vh; }
                    img { max-width:100%; max-height:100vh; }
                </style>
            </head><body>
                <img src="/stream"/>
            </body></html>''')
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            last_seq = 0
            while True:
                last_seq, frame = stream.wait_next(last_seq, timeout=1.0)
                if frame is None:
                    continue
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                except BrokenPipeError:
                    break
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        super().server_bind()


def _local_lan_ip() -> str:
    """Best-effort LAN address for the stream URL printed at startup."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(('8.8.8.8', 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return '127.0.0.1'


def start_stream_server(port):
    """Start MJPEG server; return None if port is already taken."""
    try:
        server = ReusableHTTPServer(('0.0.0.0', port), StreamHandler)
    except OSError as exc:
        if exc.errno in (48, 98):  # EADDRINUSE
            print(
                f'[MAIN] Stream port {port} busy — vision continues without MJPEG. '
                f'Stop the other process: pkill -f payload_tracker')
            return None
        raise
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ═══════════════════════════════════════════════════════════════════════════════
#  OSCILLATION ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class OscillationAnalyzer:
    def __init__(self, buffer_size=256, fft_min=0.1, fft_max=10.0,
                 damping=0.02, lowpass_alpha=0.15, csv_dir='~/payload_logs',
                 session_ts: Optional[str] = None):
        self.buf_size = buffer_size
        self.fmin = fft_min
        self.fmax = fft_max
        self.damping = damping
        self.alpha = lowpass_alpha
        self.pos_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        self.last_x = None
        self.current_freq = 0.0
        self._csv_flush_counter = 0

        csv_dir = os.path.expanduser(csv_dir)
        os.makedirs(csv_dir, exist_ok=True)
        ts = session_ts or datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(csv_dir, f'vision_osc_{ts}.csv')
        self._csv_file = open(csv_path, 'w', newline='')
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(['time_sec', 'x_raw_m', 'x_filtered_m',
                            'y_m', 'z_m', 'dominant_freq_hz', 'fps',
                            'interpolated'])
        print(f'[OSC] Logging to {csv_path}')

    def update(self, x_raw, y, z, timestamp, fps, interpolated=False):
        if self.last_x is None:
            self.last_x = x_raw
        x_filt = self.alpha * x_raw + (1.0 - self.alpha) * self.last_x
        self.last_x = x_filt
        self.pos_buffer.append(x_filt)
        self.time_buffer.append(timestamp)

        freq = self.current_freq
        if len(self.pos_buffer) == self.buf_size:
            freq = self._compute_fft()
            self.current_freq = freq

        self._csv.writerow([
            f'{timestamp:.6f}', f'{x_raw:.6f}', f'{x_filt:.6f}',
            f'{y:.6f}', f'{z:.6f}',
            f'{freq:.4f}', f'{fps:.1f}', '1' if interpolated else '0'
        ])
        self._csv_flush_counter += 1
        if self._csv_flush_counter >= 30:
            self._csv_file.flush()
            self._csv_flush_counter = 0

        return {'x_filtered': x_filt, 'freq_hz': freq,
                'damping': self.damping, 'shaper_input': [freq, self.damping]}

    def _compute_fft(self):
        signal = np.array(self.pos_buffer) - np.mean(self.pos_buffer)
        times = np.array(self.time_buffer)
        dt = float(np.mean(np.diff(times)))
        if dt <= 0:
            return self.current_freq
        N = len(signal)
        fft = np.abs(np.fft.rfft(signal * np.hanning(N)))
        freqs = np.fft.rfftfreq(N, d=dt)
        mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        if not mask.any():
            return self.current_freq
        return float(freqs[mask][np.argmax(fft[mask])])

    def close(self):
        self._csv_file.flush()
        self._csv_file.close()
        print('[OSC] CSV log closed.')


class PerTagLogger:
    """Log per-AprilTag session X, timestamp, and dX/dt."""

    def __init__(self, csv_dir='~/payload_logs', session_ts: Optional[str] = None):
        self._flush_counter = 0
        csv_dir = os.path.expanduser(csv_dir)
        os.makedirs(csv_dir, exist_ok=True)
        ts = session_ts or datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(csv_dir, f'vision_tags_{ts}.csv')
        self._csv_file = open(csv_path, 'w', newline='')
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(['time_sec', 'tag_id', 'x_m', 'dx_dt_m_s'])
        print(f'[TAGS] Logging to {csv_path}')

    def log(self, timestamp, per_tag, session_origin):
        for t in per_tag:
            rx, _ = session_origin.to_relative(t['x'], t['z'])
            self._csv.writerow([
                f'{timestamp:.6f}', t['tag_id'],
                f'{rx:.6f}', f'{t["dx_dt"]:.6f}',
            ])
        self._flush_counter += 1
        if self._flush_counter >= 30:
            self._csv_file.flush()
            self._flush_counter = 0

    def close(self):
        self._csv_file.flush()
        self._csv_file.close()
        print('[TAGS] CSV log closed.')


def _session_per_tag(per_tag, session_origin, session_time, detect_ts):
    """Session-relative X/Z and run time for each detected tag."""
    t_run = session_time.to_session(detect_ts)
    out = []
    for t in per_tag:
        rx, rz = session_origin.to_relative(t['x'], t['z'])
        out.append({
            'tag_id': t['tag_id'],
            'x': rx,
            'y': t['y'],
            'z': rz,
            'time': t_run,
            'dx_dt': t['dx_dt'],
            'dz_dt': t.get('dz_dt', 0.0),
        })
    return out


def _pair_xz_for_face(tag_states, primary_face, face_groups):
    """Session X/Z for the two tags on the active face (sorted by tag id)."""
    if not primary_face or not tag_states:
        return None
    pair_ids = sorted(face_groups.get(primary_face, ()))
    if len(pair_ids) != 2:
        return None
    by_id = {t['tag_id']: t for t in tag_states}
    t1 = by_id.get(pair_ids[0])
    t2 = by_id.get(pair_ids[1])
    if t1 is None or t2 is None:
        return None
    return (t1['x'], t1['z'], t2['x'], t2['z'])


def _pair_vel_for_face(tag_states, primary_face, face_groups):
    """Per-tag filtered velocities for the active pair (vx from dx_dt, vz from dz_dt)."""
    if not primary_face or not tag_states:
        return None
    pair_ids = sorted(face_groups.get(primary_face, ()))
    if len(pair_ids) != 2:
        return None
    by_id = {t['tag_id']: t for t in tag_states}
    t1 = by_id.get(pair_ids[0])
    t2 = by_id.get(pair_ids[1])
    if t1 is None or t2 is None:
        return None
    return (
        float(t1.get('dx_dt', 0.0)),
        float(t1.get('dz_dt', 0.0)),
        float(t2.get('dx_dt', 0.0)),
        float(t2.get('dz_dt', 0.0)),
    )


class PayloadRosPublisher:
    """ROS2 bridge: /payload/state (PayloadState) + legacy /payload/pose array."""

    POSE_LEGACY_FIELDS = (
        'time', 'x1', 'z1', 'x2', 'z2', 'vx1', 'vz1', 'vx2', 'vz2',
    )

    def __init__(
        self,
        frame_id='camera',
        wait_motion=True,
        vel_alpha=0.72,
        publish_alpha=0.14,
        publish_max_step_m=0.010,
        gantry_publish_alpha=0.16,
        publish_median=7,
        gantry_frame=True,
        gantry_yaw_deg=0.0,
        gantry_pitch_deg=0.0,
        gantry_roll_deg=0.0,
        gantry_sign=(1.0, 1.0, 1.0),
        ros_publish_hz=50.0,
        sync_adaptive=True,
    ):
        import rclpy
        from payload_perception_msgs.msg import PayloadState
        from rclpy.qos import (
            DurabilityPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )

        self._rclpy = rclpy
        self._PayloadState = PayloadState
        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node('payload_tracker')
        self._qos = qos_profile_sensor_data
        self._nan = float('nan')
        self._wait_motion = wait_motion
        self._motion_active = not wait_motion
        self._on_motion_start_cb = None
        self._vel_alpha = vel_alpha
        mw = max(3, int(publish_median))
        self._cam_smoother = PoseFilterChain(
            median_window=mw,
            ema_alpha=publish_alpha,
            max_step_m=publish_max_step_m,
        )
        self._gantry_smoother = PoseFilterChain(
            median_window=mw,
            ema_alpha=gantry_publish_alpha,
            max_step_m=publish_max_step_m * 0.75,
        )
        self._prev_motion_t = None
        self._prev_pos = None
        self._prev_vel = None
        self._prev_gantry_center = None
        self._ros_publish_hz = max(5.0, min(float(ros_publish_hz), 200.0))
        self._sync_adaptive = bool(sync_adaptive)
        self._sync_hz_ema = 30.0
        self._sync_hz_bootstrapped = False
        self._pending = None
        self._last_update_mono = 0.0
        self._ros_pub_count = 0
        self._ros_valid_count = 0
        self._ros_stats_last = time.monotonic()

        from std_msgs.msg import Float64MultiArray, MultiArrayDimension

        self._Float64MultiArray = Float64MultiArray
        self._MultiArrayDimension = MultiArrayDimension

        self.pub_state = self.node.create_publisher(
            PayloadState, '/payload/state', self._qos)
        self.pub_pose_legacy = self.node.create_publisher(
            Float64MultiArray, '/payload/pose', self._qos)

        self.frame_id = frame_id
        self._gantry_rot = None
        if gantry_frame:
            from payload_perception.payload_frames import CameraToGantryRotation
            sx, sy, sz = gantry_sign
            self._gantry_rot = CameraToGantryRotation(
                yaw_deg=gantry_yaw_deg,
                pitch_deg=gantry_pitch_deg,
                roll_deg=gantry_roll_deg,
                sign_x=sx,
                sign_y=sy,
                sign_z=sz,
            )
            self.frame_id = 'gantry'
            self.node.get_logger().info(
                'Gantry axes from camera mount (no cart position in tracker)')
        self.node.get_logger().info(
            f'Pose smoothing: cam_ema={publish_alpha:.2f} '
            f'gantry_ema={gantry_publish_alpha:.2f} '
            f'max_step={publish_max_step_m:.3f} m (median+EMA)')

        from sensor_msgs.msg import CompressedImage
        self._CompressedImage = CompressedImage
        self.pub_camera = self.node.create_publisher(
            CompressedImage, '/payload/camera/compressed', self._qos)

        if wait_motion:
            from gantry_control.msg import TrajCmd
            traj_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=10,
            )
            self._TrajCmd = TrajCmd
            self.node.create_subscription(
                TrajCmd, '/traj_cmd', self._on_traj_cmd, traj_qos)
            print('[ROS] Waiting for MOTION_START before /payload/state')
        else:
            self._TrajCmd = None
            print('[ROS] Publishing /payload/state (pose + velocity)')

        from std_msgs.msg import Empty
        self._Empty = Empty
        self.node.create_subscription(
            Empty, '/payload/reset_origin', self._on_reset_origin, 10)

        from std_msgs.msg import Float64
        self._Float64 = Float64
        sync_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.pub_sync_hz = self.node.create_publisher(
            Float64, '/stack/pose_sync_hz', sync_qos)

        self._publish_timer = None
        if self._sync_adaptive:
            self.node.get_logger().info(
                '/payload/state: adaptive (one publish per detection frame)')
        else:
            period = 1.0 / self._ros_publish_hz
            self._publish_timer = self.node.create_timer(
                period, self._on_ros_publish_timer)
            self.node.get_logger().info(
                f'/payload/state publish @ {self._ros_publish_hz:.1f} Hz '
                f'(detection may run faster)')
        self._sync_hz_timer = self.node.create_timer(
            0.5, self._on_sync_hz_timer)

    def set_motion_start_callback(self, callback):
        self._on_motion_start_cb = callback

    def _on_reset_origin(self, _msg) -> None:
        self._reset_motion_clock()
        print('[ROS] /payload/reset_origin — vision smoothers cleared')
        if self._on_motion_start_cb is not None:
            self._on_motion_start_cb()

    def _reset_motion_clock(self):
        self._prev_motion_t = None
        self._prev_pos = None
        self._prev_vel = None
        self._prev_gantry_center = None
        self._cam_smoother.reset()
        self._gantry_smoother.reset()

    def _on_traj_cmd(self, msg):
        if self._TrajCmd is None or msg.command != self._TrajCmd.MOTION_START:
            return
        self._motion_active = True
        self._reset_motion_clock()
        print('[ROS] MOTION_START — vision t=0 synced to motors')
        if self._on_motion_start_cb is not None:
            self._on_motion_start_cb()

    def _compute_vel_fallback(self, motion_t, x1, z1, x2, z2):
        """Differentiate pair position when per-tag velocities are unavailable."""
        nan = self._nan
        pos = (x1, z1, x2, z2)
        if any(not np.isfinite(v) for v in pos):
            return nan, nan, nan, nan
        if self._prev_motion_t is None or motion_t <= self._prev_motion_t:
            self._prev_motion_t = motion_t
            self._prev_pos = pos
            return nan, nan, nan, nan
        dt = motion_t - self._prev_motion_t
        if dt < 1e-4:
            return self._prev_vel or (nan, nan, nan, nan)
        p = self._prev_pos
        raw = (
            (x1 - p[0]) / dt, (z1 - p[1]) / dt,
            (x2 - p[2]) / dt, (z2 - p[3]) / dt,
        )
        if self._prev_vel is not None:
            a = self._vel_alpha
            vel = tuple(a * r + (1.0 - a) * v for r, v in zip(raw, self._prev_vel))
        else:
            vel = raw
        self._prev_motion_t = motion_t
        self._prev_pos = pos
        self._prev_vel = vel
        return vel

    def _legacy_pose_msg(
        self, motion_t, x1, z1, x2, z2, vx1, vz1, vx2, vz2,
    ):
        msg = self._Float64MultiArray()
        dim = self._MultiArrayDimension()
        dim.label = ','.join(self.POSE_LEGACY_FIELDS)
        dim.size = 9
        dim.stride = 9
        msg.layout.dim.append(dim)
        msg.layout.data_offset = 0
        msg.data = [
            float(motion_t), float(x1), float(z1), float(x2), float(z2),
            float(vx1), float(vz1), float(vx2), float(vz2),
        ]
        return msg

    def update(
        self,
        detect_ts,
        tag_states=None,
        primary_face=None,
        face_groups=None,
        valid=False,
        interpolated=False,
        center_cam=None,
    ):
        """Run smoothers on each detection frame; publish per frame if adaptive."""
        if not self._motion_active:
            return

        pair = _pair_xz_for_face(tag_states, primary_face, face_groups or {})
        if pair is not None:
            x1, z1, x2, z2 = pair
        else:
            x1 = z1 = x2 = z2 = self._nan

        vel_pair = _pair_vel_for_face(tag_states, primary_face, face_groups or {})
        if vel_pair is not None and valid and not interpolated:
            vx1, vz1, vx2, vz2 = vel_pair
        else:
            vx1, vz1, vx2, vz2 = self._compute_vel_fallback(
                detect_ts, x1, z1, x2, z2)

        nan = self._nan
        cam_x = cam_y = cam_z = nan
        gantry_x = gantry_y = gantry_z = nan
        vgx = vgy = vgz = nan

        if center_cam is not None:
            cam_x, cam_y, cam_z = self._cam_smoother.filter(
                center_cam[0], center_cam[1], center_cam[2],
                interpolated=interpolated,
            )
            if self._gantry_rot is not None:
                gx, gy, gz = self._gantry_rot.to_gantry(cam_x, cam_y, cam_z)
                gantry_x, gantry_y, gantry_z = self._gantry_smoother.filter(
                    gx, gy, gz, interpolated=interpolated,
                )
                if self._prev_gantry_center is not None:
                    t0, px, py, pz = self._prev_gantry_center
                    dt = float(detect_ts) - t0
                    if dt > 1e-4:
                        vgx = (gantry_x - px) / dt
                        vgy = (gantry_y - py) / dt
                        vgz = (gantry_z - pz) / dt
                if math.isfinite(gantry_x):
                    self._prev_gantry_center = (
                        float(detect_ts), gantry_x, gantry_y, gantry_z,
                    )

        stamp = self.node.get_clock().now().to_msg()
        state = self._PayloadState()
        state.header.stamp = stamp
        state.header.frame_id = self.frame_id
        state.motion_time_sec = float(detect_ts)
        state.x1 = float(x1)
        state.z1 = float(z1)
        state.x2 = float(x2)
        state.z2 = float(z2)
        state.vx1 = float(vx1)
        state.vz1 = float(vz1)
        state.vx2 = float(vx2)
        state.vz2 = float(vz2)
        state.cam_x = float(cam_x)
        state.cam_y = float(cam_y)
        state.cam_z = float(cam_z)
        state.gantry_x = float(gantry_x)
        state.gantry_y = float(gantry_y)
        state.gantry_z = float(gantry_z)
        state.v_gantry_x = float(vgx)
        state.v_gantry_y = float(vgy)
        state.v_gantry_z = float(vgz)
        state.valid = bool(valid)
        state.interpolated = bool(interpolated)
        state.frame_id = self.frame_id
        self._pending = {
            'state': state,
            'legacy': self._legacy_pose_msg(
                detect_ts, x1, z1, x2, z2, vx1, vz1, vx2, vz2,
            ),
            'valid': bool(valid),
        }
        self._last_update_mono = time.monotonic()
        if self._sync_adaptive:
            self._publish_pending()

    def publish(self, **kwargs):
        """Backward-compatible alias for update()."""
        self.update(**kwargs)

    def note_detect_fps(self, detect_fps: float) -> None:
        """EMA of measured detection FPS → /stack/pose_sync_hz for stack followers."""
        if detect_fps <= 0.0:
            return
        hz = max(5.0, min(float(detect_fps), 120.0))
        if not self._sync_hz_bootstrapped:
            self._sync_hz_ema = hz
            self._sync_hz_bootstrapped = True
        else:
            self._sync_hz_ema = 0.85 * self._sync_hz_ema + 0.15 * hz

    def _on_sync_hz_timer(self) -> None:
        msg = self._Float64()
        msg.data = float(self._sync_hz_ema)
        self.pub_sync_hz.publish(msg)

    def _publish_pending(self) -> None:
        if not self._motion_active or self._pending is None:
            return
        now = time.monotonic()
        rate_hz = (
            self._sync_hz_ema if self._sync_adaptive else self._ros_publish_hz)
        stale_s = 2.0 / max(rate_hz, 5.0)
        pending = self._pending
        state = pending['state']
        if now - self._last_update_mono > stale_s:
            state.interpolated = True
        self.pub_state.publish(state)
        self.pub_pose_legacy.publish(pending['legacy'])
        self._ros_pub_count += 1
        if pending.get('valid'):
            self._ros_valid_count += 1
        if now - self._ros_stats_last >= 1.0:
            dt = now - self._ros_stats_last
            hz = self._ros_pub_count / dt if dt > 0 else 0.0
            valid_pct = (
                100.0 * self._ros_valid_count / self._ros_pub_count
                if self._ros_pub_count else 0.0
            )
            self.node.get_logger().info(
                f'/payload/state ~{hz:.1f} Hz '
                f'sync={self._sync_hz_ema:.1f} valid={valid_pct:.0f}%')
            self._ros_pub_count = 0
            self._ros_valid_count = 0
            self._ros_stats_last = now

    def _on_ros_publish_timer(self):
        self._publish_pending()

    def publish_camera(self, jpeg_bytes: bytes) -> None:
        """Annotated HD camera frame for mission planner (not gated on MOTION_START)."""
        msg = self._CompressedImage()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'
        msg.data = bytes(jpeg_bytes)
        self.pub_camera.publish(msg)

    def spin_once(self):
        self._rclpy.spin_once(self.node, timeout_sec=0)

    def shutdown(self):
        if self._publish_timer is not None:
            self._publish_timer.cancel()
        if hasattr(self, '_sync_hz_timer'):
            self._sync_hz_timer.cancel()
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
#  APRILTAG TRACKER (active face pair only)
# ═══════════════════════════════════════════════════════════════════════════════

class ArucoTracker:
    """Track payload center from whichever tagged cube faces are visible."""

    _BOX_HALF_SIZE = 0.100 / 2.0
    _FACE_TAG_CENTERS_M = (
        (0.050, 0.080),  # top tag: base + 0
        (0.020, 0.020),  # left/bottom-left tag: base + 1
        (0.080, 0.020),  # right/bottom-right tag: base + 2
    )

    _R_FRONT = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    _R_LEFT = np.eye(3, dtype=np.float64)
    _R_RIGHT = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    _R_BACK = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)

    @classmethod
    def _build_tag_registry(cls):
        h = cls._BOX_HALF_SIZE
        centers = cls._FACE_TAG_CENTERS_M

        def centered(face_u: float, face_v: float) -> tuple[float, float]:
            return face_u - h, face_v - h

        registry = {}
        face_defs = [
            ('front', 0, cls._R_FRONT),
            ('right', 3, cls._R_RIGHT),
            ('back', 6, cls._R_BACK),
            ('left', 9, cls._R_LEFT),
        ]
        for face, base_id, rot in face_defs:
            for idx, (u, v) in enumerate(centers):
                a, y = centered(u, v)
                if face == 'front':
                    pos = np.array([h, y, a])
                elif face == 'right':
                    pos = np.array([-a, y, -h])
                elif face == 'back':
                    pos = np.array([-h, y, -a])
                else:  # left
                    pos = np.array([a, y, h])
                registry[base_id + idx] = (pos, rot, face)
        return registry

    _FACE_NAME_MAP = {
        'plus_x': 'front', 'minus_x': 'back',
        'minus_z': 'right', 'plus_z': 'left',
    }
    _FACE_ROTATION_MAP = {
        'front': _R_FRONT, 'back': _R_BACK,
        'right': _R_RIGHT, 'left': _R_LEFT,
    }

    @classmethod
    def _build_tag_registry_from_json(cls, layout_path: str):
        with open(layout_path, 'r') as f:
            data = json.load(f)
        registry = {}
        face_groups: dict[str, set[int]] = {}
        for tag in data['tags']:
            tid = int(tag['id'])
            face_key = tag['face']
            face_name = cls._FACE_NAME_MAP.get(face_key, face_key)
            rot = cls._FACE_ROTATION_MAP.get(face_name, np.eye(3, dtype=np.float64))
            center = np.array(tag['center_m'], dtype=np.float64)
            registry[tid] = (center, rot, face_name)
            face_groups.setdefault(face_name, set()).add(tid)
        return registry, face_groups

    def __init__(self, marker_size=0.10, tag_family='tagStandard41h12',
                 decision_margin_min=12.0, nthreads=4, quad_decimate=2.0,
                 aruco_dict=None, debug_detect=False, rigid_body_pnp=True,
                 rigid_reproj_max_px=8.0, layout_path=None):
        if aruco_dict and not tag_family:
            tag_family = aruco_dict
        self.marker_size = marker_size
        self.decision_margin_min = decision_margin_min
        if layout_path is not None:
            self.tag_registry, self.face_groups = self._build_tag_registry_from_json(layout_path)
        else:
            self.tag_registry = self._build_tag_registry()
            self.face_groups = {
                'front': {0, 1, 2},
                'right': {3, 4, 5},
                'back': {6, 7, 8},
                'left': {9, 10, 11},
            }
        self.valid_ids = set(self.tag_registry.keys())
        self.detector = Detector(
            families=tag_family, nthreads=nthreads,
            quad_decimate=quad_decimate, refine_edges=1,
        )
        self._map1 = None
        self._map2 = None
        self._intrinsics_key = None
        self._debug_detect = bool(debug_detect)
        self._rigid_body_pnp = bool(rigid_body_pnp)
        self._rigid_reproj_max_px = float(rigid_reproj_max_px)
        self._dbg_last = 0.0
        self._prev_center = {}
        self._prev_center_vel = {}
        self._prev_tag_x = {}
        self._prev_tag_x_vel = {}
        self._prev_tag_z = {}
        self._prev_tag_z_vel = {}
        self._tag_z_vel_alpha = 0.72
        self._center_vel_alpha = 0.72
        self._tag_x_vel_alpha = 0.72
        # Last good corners per tag (keeps pair visible through brief dropout).
        self._tag_snapshots: dict[int, dict] = {}

        self.FACE_CONFIRM_FRAMES = 3
        self.FACE_DROP_FRAMES = 12
        self._face_present_count: dict[str, int] = {}
        self._face_absent_count: dict[str, int] = {}
        self._committed_faces: set[str] = set()
        self._primary_face: Optional[str] = None
        self._remap_interp = cv2.INTER_LINEAR

    def _ensure_undistort_maps(self, gray, camera_matrix, dist_coeffs):
        key = (gray.shape, camera_matrix.tobytes(), dist_coeffs.tobytes())
        if self._intrinsics_key == key:
            return
        h, w = gray.shape
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix,
            (w, h), cv2.CV_16SC2,
        )
        self._intrinsics_key = key

    def _update_committed_faces(self, currently_visible: set[str]) -> set[str]:
        for face in ('front', 'right', 'back', 'left'):
            if face in currently_visible:
                self._face_present_count[face] = (
                    self._face_present_count.get(face, 0) + 1
                )
                self._face_absent_count[face] = 0
                if self._face_present_count[face] >= self.FACE_CONFIRM_FRAMES:
                    self._committed_faces.add(face)
            else:
                self._face_absent_count[face] = (
                    self._face_absent_count.get(face, 0) + 1
                )
                self._face_present_count[face] = 0
                if self._face_absent_count[face] >= self.FACE_DROP_FRAMES:
                    self._committed_faces.discard(face)
                    if self._primary_face == face:
                        self._primary_face = None
        return self._committed_faces

    @staticmethod
    def _group_by_face(detections, tag_registry):
        by_face: dict[str, list] = {}
        for d in detections:
            face = tag_registry[d.tag_id][2]
            by_face.setdefault(face, []).append(d)
        return by_face

    def _select_primary_face(self, by_face: dict[str, list], committed: set[str]):
        candidates = []
        for face in committed:
            dets = by_face.get(face, [])
            if not dets:
                continue
            score = sum(d.decision_margin for d in dets) + 100.0 * len(dets)
            candidates.append((score, face, dets))
        if not candidates:
            return None, []
        candidates.sort(reverse=True)
        if self._primary_face is not None:
            for _, face, dets in candidates:
                if face == self._primary_face:
                    return face, dets
        self._primary_face = candidates[0][1]
        return candidates[0][1], candidates[0][2]

    def _payload_position(self, d):
        t_tag_p, R_tag_p, _ = self.tag_registry[d.tag_id]
        R_pc = d.pose_R @ R_tag_p.T
        return d.pose_t.flatten() - R_pc @ t_tag_p

    def _tag_object_corners_payload(self, tag_id: int):
        """AprilTag corner points in payload/body coordinates."""
        t_tag_p, R_tag_p, _ = self.tag_registry[tag_id]
        h = 0.5 * float(self.marker_size)
        tag_corners = np.array(
            [
                [-h, h, 0.0],
                [h, h, 0.0],
                [h, -h, 0.0],
                [-h, -h, 0.0],
            ],
            dtype=np.float64,
        )
        return t_tag_p.reshape(1, 3) + tag_corners @ R_tag_p.T

    def _position_from_rigid_pnp(self, detections, camera_matrix):
        if not self._rigid_body_pnp or not detections:
            return None

        object_points = []
        image_points = []
        for d in detections:
            object_points.append(self._tag_object_corners_payload(d.tag_id))
            image_points.append(np.asarray(d.corners, dtype=np.float64))

        if not object_points:
            return None

        obj = np.vstack(object_points).reshape(-1, 3)
        img = np.vstack(image_points).reshape(-1, 2)
        if obj.shape[0] < 4:
            return None

        flags = cv2.SOLVEPNP_ITERATIVE
        if obj.shape[0] == 4:
            flags = cv2.SOLVEPNP_IPPE

        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            camera_matrix,
            None,
            flags=flags,
        )
        if not ok:
            return None

        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, None)
        projected = projected.reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=1))))
        if not math.isfinite(err) or err > self._rigid_reproj_max_px:
            return None

        t = tvec.flatten()
        return float(t[0]), float(t[1]), float(t[2]), err

    def _position_from_detections(self, detections):
        t_estimates = [self._payload_position(d) for d in detections]
        t_mean = np.mean(t_estimates, axis=0)
        return float(t_mean[0]), float(t_mean[1]), float(t_mean[2])

    def _per_tag_positions(self, detections, ts):
        """Body-frame X per tag plus smoothed dX/dt (m/s)."""
        out = []
        for d in detections:
            t_pc = self._payload_position(d)
            x = float(t_pc[0])
            y = float(t_pc[1])
            z = float(t_pc[2])
            dx_dt = 0.0
            dz_dt = 0.0
            if d.tag_id in self._prev_tag_x:
                prev_ts, prev_x = self._prev_tag_x[d.tag_id]
                dt = ts - prev_ts
                if dt > 1e-4:
                    dx_dt = (x - prev_x) / dt
            if d.tag_id in self._prev_tag_x_vel:
                a = self._tag_x_vel_alpha
                dx_dt = a * dx_dt + (1.0 - a) * self._prev_tag_x_vel[d.tag_id]
            if d.tag_id in self._prev_tag_z:
                prev_ts, prev_z = self._prev_tag_z[d.tag_id]
                dt = ts - prev_ts
                if dt > 1e-4:
                    dz_dt = (z - prev_z) / dt
            if d.tag_id in self._prev_tag_z_vel:
                a = self._tag_z_vel_alpha
                dz_dt = a * dz_dt + (1.0 - a) * self._prev_tag_z_vel[d.tag_id]
            self._prev_tag_x[d.tag_id] = (ts, x)
            self._prev_tag_x_vel[d.tag_id] = dx_dt
            self._prev_tag_z[d.tag_id] = (ts, z)
            self._prev_tag_z_vel[d.tag_id] = dz_dt
            out.append({
                'tag_id': d.tag_id,
                'x': x,
                'y': y,
                'z': z,
                'dx_dt': float(dx_dt),
                'dz_dt': float(dz_dt),
            })
        return out

    def _update_tag_snapshots(self, detections, ts):
        """Per-tag snapshot with rigid center velocity (preserves square shape)."""
        for d in detections:
            corners = d.corners.astype(np.float32)
            center = corners.mean(axis=0)
            center_vel = np.zeros(2, dtype=np.float32)
            if d.tag_id in self._prev_center:
                prev_ts, prev_c = self._prev_center[d.tag_id]
                dt = ts - prev_ts
                if dt > 1e-4:
                    center_vel = ((center - prev_c) / dt).astype(np.float32)
            if d.tag_id in self._prev_center_vel:
                a = self._center_vel_alpha
                center_vel = (
                    a * center_vel + (1.0 - a) * self._prev_center_vel[d.tag_id]
                ).astype(np.float32)
            self._prev_center[d.tag_id] = (ts, center.copy())
            self._prev_center_vel[d.tag_id] = center_vel
            d._center_vel = center_vel
            self._tag_snapshots[d.tag_id] = {
                'tag_id': d.tag_id,
                'corners': corners.copy(),
                'center_vel': center_vel.copy(),
                'ts': ts,
            }

    def tags_for_overlay(self, primary_face, live_detections, now_ts, hold_sec):
        """Live detections plus held snapshots so a pair stays complete."""
        by_id = {d.tag_id: d for d in live_detections}
        if not primary_face or primary_face not in self.face_groups:
            return list(live_detections)
        out = []
        for tid in sorted(self.face_groups[primary_face]):
            if tid in by_id:
                out.append(by_id[tid])
                continue
            snap = self._tag_snapshots.get(tid)
            if snap is not None and (now_ts - snap['ts']) <= hold_sec:
                out.append(snap)
        return out

    def detect(self, gray, camera_matrix, dist_coeffs, device_timestamp=None):
        self._ensure_undistort_maps(gray, camera_matrix, dist_coeffs)
        undist = cv2.remap(
            gray, self._map1, self._map2, self._remap_interp,
        )

        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])

        with _SuppressStderr():
            detections = self.detector.detect(
                undist, estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy), tag_size=self.marker_size,
            )

        faces_seen = set()
        raw_used = []
        for d in detections:
            if d.tag_id not in self.valid_ids:
                continue
            if d.decision_margin < self.decision_margin_min:
                continue
            faces_seen.add(self.tag_registry[d.tag_id][2])
            raw_used.append(d)

        ts = device_timestamp if device_timestamp is not None else time.monotonic()
        self._update_tag_snapshots(raw_used, ts)

        committed = self._update_committed_faces(faces_seen)
        by_face = self._group_by_face(raw_used, self.tag_registry)
        select_faces = committed if committed else set(by_face.keys())
        primary_face, used = self._select_primary_face(by_face, select_faces)
        # Keep every tag on the primary face that was detected this frame.
        if primary_face and primary_face in by_face:
            used = by_face[primary_face]

        if self._debug_detect:
            now = time.monotonic()
            if now - self._dbg_last > 1.0:
                ids_seen = [d.tag_id for d in detections]
                print(
                    f'[DETECT-DBG] ids={ids_seen} committed={sorted(committed)} '
                    f'primary={primary_face} n_used={len(used)}'
                )
                self._dbg_last = now

        per_tag = self._per_tag_positions(raw_used, ts)

        if not used:
            return {
                'per_tag': per_tag,
                'detect_timestamp': ts,
            }

        rigid = self._position_from_rigid_pnp(raw_used, camera_matrix)
        if rigid is not None:
            X, Y, Z, reproj_err = rigid
            pose_source = 'rigid_pnp'
        else:
            fallback_detections = raw_used if raw_used else used
            X, Y, Z = self._position_from_detections(fallback_detections)
            reproj_err = float('nan')
            pose_source = 'tag_average_all'

        return {
            'X': X, 'Y': Y, 'Z': Z,
            'primary_face': primary_face,
            'face_detections': used,
            'detect_timestamp': ts,
            'per_tag': per_tag,
            'pose_source': pose_source,
            'reproj_err_px': reproj_err,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  POSITION INTERPOLATOR + TOP-DOWN
# ═══════════════════════════════════════════════════════════════════════════════

class SessionOrigin:
    """Session zero at first detection (camera X, Y, Z)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._ox = None
        self._oy = None
        self._oz = None

    def to_relative(self, x, z):
        """Legacy: horizontal camera X,Z only."""
        rx, _, rz = self.to_relative_xyz(x, 0.0, z)
        return rx, rz

    def to_relative_xyz(self, x, y, z):
        if self._ox is None:
            self._ox = float(x)
            self._oy = float(y)
            self._oz = float(z)
        return (
            float(x) - self._ox,
            float(y) - self._oy,
            float(z) - self._oz,
        )


class SessionTime:
    """Each run starts at t=0; first frame timestamp sets the session clock."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._t0 = None

    def to_session(self, t_sec):
        t = float(t_sec)
        if self._t0 is None:
            self._t0 = t
        return t - self._t0


class PositionSmoother:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self._x = 0.0
        self._z = 0.0

    def update(self, x, z):
        self._x = self.alpha * x + (1.0 - self.alpha) * self._x
        self._z = self.alpha * z + (1.0 - self.alpha) * self._z
        return self._x, self._z


class PoseFilterChain:
    """
    Median-of-recent-window then EMA, with per-frame step clamp.
    Used on camera center and again on gantry axes after rotation.
    """

    def __init__(
        self,
        *,
        median_window: int = 7,
        ema_alpha: float = 0.14,
        max_step_m: float = 0.010,
        interp_alpha: float = 0.08,
    ):
        self._median_window = max(3, int(median_window))
        self.ema_alpha = float(ema_alpha)
        self.interp_alpha = float(interp_alpha)
        self.max_step_m = float(max_step_m)
        self._buf: deque = deque(maxlen=self._median_window)
        self._p: Optional[np.ndarray] = None

    def reset(self):
        self._buf.clear()
        self._p = None

    def filter(
        self, x: float, y: float, z: float, *, interpolated: bool = False,
    ) -> Tuple[float, float, float]:
        self._buf.append([float(x), float(y), float(z)])
        if len(self._buf) >= 3:
            med = np.median(np.array(self._buf, dtype=float), axis=0)
        else:
            med = np.array([float(x), float(y), float(z)], dtype=float)

        if self._p is None:
            self._p = med.copy()
            return float(self._p[0]), float(self._p[1]), float(self._p[2])

        step = med - self._p
        mx = float(np.max(np.abs(step)))
        if mx > self.max_step_m:
            med = self._p + step * (self.max_step_m / mx)

        a = self.interp_alpha if interpolated else self.ema_alpha
        self._p = a * med + (1.0 - a) * self._p
        return float(self._p[0]), float(self._p[1]), float(self._p[2])


class PositionInterpolator:
    def __init__(self, max_gap_sec=0.5):
        self.max_gap = max_gap_sec
        self.history = deque(maxlen=2)
        self.last_real_time = None

    def add_real(self, timestamp, X, Y, Z):
        self.history.append((timestamp, X, Y, Z))
        self.last_real_time = timestamp

    def predict(self, timestamp):
        if self.last_real_time is None:
            return None
        if timestamp - self.last_real_time > self.max_gap:
            return None
        if len(self.history) < 2:
            return self.history[-1][1:]
        t0, x0, y0, z0 = self.history[0]
        t1, x1, y1, z1 = self.history[1]
        dt = t1 - t0
        if dt <= 0:
            return (x1, y1, z1)
        a = (timestamp - t1) / dt
        return (
            x1 + a * (x1 - x0),
            y1 + a * (y1 - y0),
            z1 + a * (z1 - z0),
        )


def _display_dt(hd_ts, det_ts, held, hold_sec, lead_frames=0.0):
    """Forward delta to paint overlays at the HD frame time."""
    if lead_frames <= 0:
        return float(np.clip(max(0.0, hd_ts - det_ts), 0.0, hold_sec if held else 0.12))
    lead = lead_frames / 30.0  # fallback if ever re-enabled
    dt = max(0.0, hd_ts - det_ts) + lead
    cap = hold_sec if held else 0.28
    return float(np.clip(dt, 0.0, cap))


def _rigid_corners_hd(corners_lr, center_vel, dt_display, scale_x, scale_y):
    """Move tag as a rigid square: translate center only, keep corner offsets."""
    corners = np.asarray(corners_lr, dtype=np.float32)
    center = corners.mean(axis=0)
    offsets = corners - center
    if center_vel is not None and dt_display > 0:
        center = center + center_vel * dt_display
    hd = center + offsets
    hd[:, 0] *= scale_x
    hd[:, 1] *= scale_y
    return hd


def _draw_overlay(hd_view, tags_to_draw, scale_x, scale_y, dt_display, held):
    """Draw tag outlines + midpoint. tags_to_draw: detections or held snapshots."""
    color = (0, 200, 255) if held else (0, 255, 0)
    centers = []

    for item in tags_to_draw:
        if hasattr(item, 'corners'):
            corners = item.corners
            vel = getattr(item, '_center_vel', None)
        else:
            corners = item['corners']
            vel = item.get('center_vel')
        hd = _rigid_corners_hd(corners, vel, dt_display, scale_x, scale_y)
        cv2.polylines(hd_view, [hd.astype(np.int32)], True, color, 2)
        centers.append((hd[:, 0].mean(), hd[:, 1].mean()))

    if len(centers) >= 2:
        cx = int(round((centers[0][0] + centers[1][0]) / 2))
        cy = int(round((centers[0][1] + centers[1][1]) / 2))
    elif centers:
        cx, cy = int(round(centers[0][0])), int(round(centers[0][1]))
    else:
        return

    cv2.circle(hd_view, (cx, cy), 10, (0, 255, 255), -1)
    cv2.circle(hd_view, (cx, cy), 16, (0, 255, 255), 2)


def _draw_per_tag_labels(hd_view, tags_to_draw, tag_states, scale_x, scale_y,
                         dt_display, held):
    """Label each tag with ID, time, session X and Z."""
    if not tag_states:
        return
    by_id = {t['tag_id']: t for t in tag_states}
    color = (0, 200, 255) if held else (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for item in tags_to_draw:
        if hasattr(item, 'tag_id'):
            tid = item.tag_id
            corners = item.corners
            vel = getattr(item, '_center_vel', None)
        else:
            tid = item['tag_id']
            corners = item['corners']
            vel = item.get('center_vel')
        state = by_id.get(tid)
        if state is None:
            continue

        hd = _rigid_corners_hd(corners, vel, dt_display, scale_x, scale_y)
        cx = int(hd[:, 0].mean())
        cy = int(hd[:, 1].mean())
        label = (
            f'ID{tid} t={state["time"]:.3f}s '
            f'X={state["x"]:+.3f} Z={state["z"]:.3f}'
        )
        ty = max(18, cy - 12)
        cv2.putText(hd_view, label, (cx - 80, ty), font, 0.45, (0, 0, 0), 3)
        cv2.putText(hd_view, label, (cx - 80, ty), font, 0.45, color, 1)

    # Side panel listing all tags (stable layout when tags move)
    panel_x, panel_y = 12, 88
    line_h = 22
    n_lines = len(tag_states) + 1
    cv2.rectangle(
        hd_view,
        (panel_x - 6, panel_y - 20),
        (panel_x + 420, panel_y + n_lines * line_h + 4),
        (0, 0, 0), -1,
    )
    cv2.putText(
        hd_view, 'AprilTags (session X/Z, run time from 0):',
        (panel_x, panel_y - 4), font, 0.5, (200, 200, 200), 1,
    )
    for i, t in enumerate(sorted(tag_states, key=lambda s: s['tag_id'])):
        line = (
            f'  ID{t["tag_id"]}  t={t["time"]:.3f}s  '
            f'X={t["x"]:+.3f}m  Z={t["z"]:.3f}m'
        )
        cv2.putText(
            hd_view, line,
            (panel_x, panel_y + i * line_h + 16), font, 0.48, color, 1,
        )


class TopDownView:
    """Session X/Z plot (Z up). Auto-scales; always shows origin (0, 0)."""

    MIN_SPAN = 0.28
    ZERO_EXTENT = 0.18
    PIXEL_MARGIN = 22

    def __init__(self, width, height, x_range=(-1.0, 1.0), z_range=(-0.6, 1.2)):
        self.width = width
        self.height = height
        self._x_lim = x_range
        self._z_lim = z_range
        self.trail = deque(maxlen=60)

    def _compute_bounds(self, marker_xz=None):
        xs = [0.0]
        zs = [0.0]
        if marker_xz is not None:
            xs.append(float(marker_xz[0]))
            zs.append(float(marker_xz[1]))
        for x, z in self.trail:
            xs.append(float(x))
            zs.append(float(z))

        x_min = min(min(xs), -self.ZERO_EXTENT, self._x_lim[0])
        x_max = max(max(xs), self.ZERO_EXTENT, self._x_lim[1])
        z_min = min(min(zs), -self.ZERO_EXTENT, self._z_lim[0])
        z_max = max(max(zs), self.ZERO_EXTENT, self._z_lim[1])

        x_span = max(self.MIN_SPAN, x_max - x_min)
        z_span = max(self.MIN_SPAN, z_max - z_min)
        if x_max - x_min < self.MIN_SPAN:
            xc = 0.5 * (x_min + x_max)
            x_min, x_max = xc - 0.5 * self.MIN_SPAN, xc + 0.5 * self.MIN_SPAN
        if z_max - z_min < self.MIN_SPAN:
            zc = 0.5 * (z_min + z_max)
            z_min, z_max = zc - 0.5 * self.MIN_SPAN, zc + 0.5 * self.MIN_SPAN
            z_span = self.MIN_SPAN

        x_pad = max(0.08, (x_max - x_min) * 0.12)
        z_pad_top = max(0.06, z_span * 0.12)
        z_pad_bot = max(0.14, z_span * 0.38)
        return (
            x_min - x_pad, x_max + x_pad,
            z_min - z_pad_bot, z_max + z_pad_top,
        )

    def world_to_pixel(self, X, Z, bounds):
        x_min, x_max, z_min, z_max = bounds
        m = self.PIXEL_MARGIN
        pw = max(1, self.width - 2 * m)
        ph = max(1, self.height - 2 * m)
        x_d = max(1e-9, x_max - x_min)
        z_d = max(1e-9, z_max - z_min)
        px = int(round(m + (float(X) - x_min) / x_d * pw))
        py = int(round(m + (1.0 - (float(Z) - z_min) / z_d) * ph))
        px = int(np.clip(px, m, self.width - m - 1))
        py = int(np.clip(py, m, self.height - m - 1))
        return px, py

    def render(self, marker_xz=None, interpolated=False, update_trail=True):
        bounds = self._compute_bounds(marker_xz)
        img = np.full((self.height, self.width, 3), (25, 25, 30), dtype=np.uint8)
        grid_color = (55, 55, 65)
        x_min, x_max, z_min, z_max = bounds

        for x_grid in np.arange(
            math.floor(x_min * 2) / 2, x_max + 0.01, 0.25,
        ):
            px, _ = self.world_to_pixel(x_grid, 0, bounds)
            cv2.line(img, (px, 0), (px, self.height), grid_color, 1)
        for z_grid in np.arange(
            math.floor(z_min * 2) / 2, z_max + 0.01, 0.25,
        ):
            _, py = self.world_to_pixel(0, z_grid, bounds)
            cv2.line(img, (0, py), (self.width, py), grid_color, 1)

        ox, oy = self.world_to_pixel(0, 0, bounds)
        axis_color = (90, 110, 140)
        cv2.line(img, (ox, 0), (ox, self.height), axis_color, 1)
        cv2.line(img, (0, oy), (self.width, oy), axis_color, 1)
        cv2.drawMarker(img, (ox, oy), (210, 210, 220),
                       cv2.MARKER_CROSS, 12, 2)
        cv2.putText(img, '0', (ox + 4, oy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 210), 1)

        if marker_xz is not None and update_trail:
            self.trail.append(marker_xz)
        n = len(self.trail)
        for i in range(1, n):
            p1 = self.world_to_pixel(*self.trail[i - 1], bounds)
            p2 = self.world_to_pixel(*self.trail[i], bounds)
            a = i / n
            cv2.line(img, p1, p2, (int(50 * a), int(255 * a), int(100 * a)), 2)

        if marker_xz is not None:
            px, py = self.world_to_pixel(*marker_xz, bounds)
            color = (0, 165, 255) if interpolated else (0, 255, 0)
            cv2.circle(img, (px, py), 8, color, -1)
            cv2.circle(img, (px, py), 14, color, 2)
            cv2.putText(img, f'X={marker_xz[0]:+.3f}m  Z={marker_xz[1]:.3f}m',
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(img, 'TOP-DOWN (X right, Z up)', (8, self.height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        return img


class StreamEncoder(threading.Thread):
    def __init__(self, quality=75, jpeg_queue=None, use_http=False):
        super().__init__(daemon=True)
        self._frame = None
        self._new = False
        self._lock = threading.Lock()
        self._event = threading.Event()
        self.quality = quality
        self._jpeg_queue = jpeg_queue
        self._use_http = use_http

    def submit(self, frame):
        with self._lock:
            self._frame = frame.copy()
            self._new = True
        self._event.set()

    def run(self):
        while True:
            self._event.wait()
            self._event.clear()
            with self._lock:
                if not self._new:
                    continue
                frame = self._frame
                self._new = False
            ok, buf = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                continue
            jpeg = buf.tobytes()
            if self._jpeg_queue is not None:
                try:
                    self._jpeg_queue.put_nowait(jpeg)
                except queue.Full:
                    try:
                        self._jpeg_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._jpeg_queue.put_nowait(jpeg)
                    except queue.Full:
                        pass
            if self._use_http:
                stream.update(frame, self.quality)


def _drain_queue(queue):
    msg = None
    while True:
        m = queue.tryGet()
        if m is None:
            break
        msg = m
    return msg


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _argv_without_ros_launch(argv):
    """Drop args appended by ros2 launch Node() (--ros-args ...)."""
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    return argv


def main():
    parser = argparse.ArgumentParser(
        description='Payload tracker — decoupled detection + HD display')
    parser.add_argument('--ip', type=str, default=None)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=400)
    parser.add_argument('--fps', type=int, default=128,
                        help='Detection camera FPS (sensor max ~129)')
    parser.add_argument('--quad-decimate', type=float, default=2.0,
                        help='AprilTag decimation (2.5 = faster, 2.0 = more accurate)')
    parser.add_argument('--turbo', action='store_true',
                        help='Faster path: coarser tags, smaller top-down, lower JPEG cost')
    parser.add_argument('--display-width', type=int, default=1280)
    parser.add_argument('--display-height', type=int, default=800)
    parser.add_argument('--display-fps', type=int, default=30,
                        help='HD stream FPS (lower frees CPU for detection)')
    parser.add_argument('--topdown-size', type=int, default=400,
                        help='Top-down panel edge length in pixels')
    parser.add_argument('--jpeg-quality', type=int, default=75)
    parser.add_argument('--marker-size', type=float, default=0.10)
    parser.add_argument('--aruco-dict', type=str, default='tagStandard41h12')
    parser.add_argument('--max-interp', type=float, default=0.5)
    parser.add_argument('--stream-port', type=int, default=8080)
    parser.add_argument('--no-stream', action='store_true')
    parser.add_argument(
        '--stream-detect-only', action='store_true',
        help='Use annotated detection frames for the web/ROS camera stream; '
             'do not request a separate HD display output from the OAK.',
    )
    parser.add_argument('--ros', action='store_true',
                        help='Publish poses on /payload/* ROS2 topics')
    parser.add_argument(
        '--wait-motion', action='store_true', default=None,
        help='Publish /payload/state only after /traj_cmd MOTION_START (default with --ros)',
    )
    parser.add_argument(
        '--no-wait-motion', action='store_true',
        help='Publish /payload/state immediately (legacy behavior)',
    )
    parser.add_argument('--frame-id', type=str, default='camera',
                        help='frame_id when --no-gantry-frame (default camera)')
    parser.add_argument('--no-gantry-frame', action='store_true',
                        help='Keep /payload/state in camera axes only (no gantry_x/y/z)')
    parser.add_argument('--gantry-yaw-deg', type=float, default=0.0)
    parser.add_argument('--gantry-pitch-deg', type=float, default=0.0)
    parser.add_argument('--gantry-roll-deg', type=float, default=0.0)
    parser.add_argument('--gantry-sign-x', type=float, default=1.0)
    parser.add_argument('--gantry-sign-y', type=float, default=1.0)
    parser.add_argument('--gantry-sign-z', type=float, default=1.0)
    parser.add_argument('--buf-size', type=int, default=256)
    parser.add_argument('--fft-min', type=float, default=0.1)
    parser.add_argument('--fft-max', type=float, default=10.0)
    parser.add_argument('--damping', type=float, default=0.02)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--pos-alpha', type=float, default=0.40,
                        help='EMA for on-screen overlay top-down (default 0.40)')
    parser.add_argument('--publish-alpha', type=float, default=0.14,
                        help='Camera-center EMA after median (lower = smoother)')
    parser.add_argument('--gantry-publish-alpha', type=float, default=0.16,
                        help='Second EMA on gantry_x/y/z after mount rotation')
    parser.add_argument('--publish-max-step-m', type=float, default=0.010,
                        help='Max m per detect frame toward EMA (spike reject)')
    parser.add_argument('--publish-median', type=int, default=7,
                        help='Median window length (odd, >=3)')
    parser.add_argument('--ros-publish-hz', type=float, default=50.0,
                        help='Fixed /payload/state rate when --no-sync-adaptive')
    parser.add_argument(
        '--sync-adaptive', action='store_true', default=None,
        help='Publish /payload/state per detection; drive /stack/pose_sync_hz (default with --ros)',
    )
    parser.add_argument(
        '--no-sync-adaptive', action='store_true',
        help='Use fixed --ros-publish-hz timer instead of per-frame publish',
    )
    parser.add_argument(
        '--camera-jpeg-fps', type=float, default=15.0,
        help='ROS /payload/camera/compressed rate when HD stream is off',
    )
    parser.add_argument('--decision-margin-min', type=float, default=12.0,
                        help='Minimum AprilTag decision margin')
    parser.add_argument('--hold-sec', type=float, default=0.35)
    parser.add_argument('--display-lead', type=float, default=0.0,
                        help='HD-frame lead for overlay extrapolation (0 = no lookahead)')
    parser.add_argument(
        '--debug-detect', action='store_true',
        help='Log AprilTag face selection once per second ([DETECT-DBG])',
    )
    parser.add_argument(
        '--no-rigid-body-pnp', action='store_true',
        help='Use old per-tag center averaging instead of rigid multi-tag solvePnP',
    )
    parser.add_argument(
        '--rigid-reproj-max-px', type=float, default=8.0,
        help='Fallback to tag averaging if rigid solve RMS reprojection exceeds this',
    )
    parser.add_argument('--layout', type=str, default=None,
                        help='Path to tag_layout JSON (overrides hardcoded face mapping)')
    parser.add_argument('--csv-dir', type=str, default='~/payload_logs')
    parser.add_argument('--x-range', type=float, default=1.0,
                        help='Minimum half-span for top-down X (m); auto-zoom includes 0')
    parser.add_argument('--z-min', type=float, default=-0.6,
                        help='Top-down Z axis minimum (m); session zero always shown')
    parser.add_argument('--z-max', type=float, default=1.2,
                        help='Top-down Z axis maximum (m)')
    parser.add_argument('--z-range', type=float, default=None,
                        help='Deprecated: use --z-max; if set, z_min=-z_range*0.2 z_max=z_range')
    args = parser.parse_args(_argv_without_ros_launch(sys.argv[1:]))

    if args.no_wait_motion:
        args.wait_motion = False
    elif args.wait_motion is None:
        args.wait_motion = bool(args.ros)

    if args.no_sync_adaptive:
        args.sync_adaptive = False
    elif args.sync_adaptive is None:
        args.sync_adaptive = bool(args.ros)

    args.gantry_frame = bool(args.ros) and not args.no_gantry_frame

    if args.z_range is not None:
        args.z_max = float(args.z_range)
        args.z_min = -0.2 * args.z_max
    else:
        args.z_min = float(args.z_min)
        args.z_max = float(args.z_max)

    if args.turbo:
        args.quad_decimate = max(args.quad_decimate, 2.5)
        args.topdown_size = min(args.topdown_size, 320)
        args.jpeg_quality = min(args.jpeg_quality, 65)

    if args.ip:
        prev = os.environ.get('DEPTHAI_DEVICE_NAME')
        if prev and prev != args.ip:
            print(f'[MAIN] OAK IP override: {prev} → {args.ip}')
        os.environ['DEPTHAI_DEVICE_NAME'] = args.ip
        print(f'[MAIN] Connecting to OAK camera at {args.ip}')
    elif os.environ.get('DEPTHAI_DEVICE_NAME'):
        print(f'[MAIN] OAK from env DEPTHAI_DEVICE_NAME='
              f'{os.environ["DEPTHAI_DEVICE_NAME"]}')
    else:
        print('[MAIN] OAK: auto-discover (pass --ip if connection fails)')

    tracker = ArucoTracker(
        tag_family=args.aruco_dict,
        marker_size=args.marker_size,
        quad_decimate=args.quad_decimate,
        decision_margin_min=args.decision_margin_min,
        debug_detect=args.debug_detect,
        rigid_body_pnp=not args.no_rigid_body_pnp,
        rigid_reproj_max_px=args.rigid_reproj_max_px,
        layout_path=args.layout,
    )
    if args.turbo:
        tracker._remap_interp = cv2.INTER_NEAREST
    session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    osc = OscillationAnalyzer(
        buffer_size=args.buf_size, fft_min=args.fft_min,
        fft_max=args.fft_max, damping=args.damping,
        lowpass_alpha=args.alpha, csv_dir=args.csv_dir,
        session_ts=session_ts,
    )
    tag_logger = PerTagLogger(csv_dir=args.csv_dir, session_ts=session_ts)
    interpolator = PositionInterpolator(max_gap_sec=args.max_interp)
    pos_smoother = PositionSmoother(alpha=args.pos_alpha)
    world_smoother = PoseFilterChain(
        median_window=max(3, args.publish_median),
        ema_alpha=max(0.08, args.publish_alpha * 1.1),
        max_step_m=args.publish_max_step_m * 1.5,
    )
    session_origin = SessionOrigin()
    session_time = SessionTime()

    ros_pub = None
    if args.ros:
        try:
            ros_pub = PayloadRosPublisher(
                frame_id=args.frame_id,
                wait_motion=args.wait_motion,
                publish_alpha=args.publish_alpha,
                publish_max_step_m=args.publish_max_step_m,
                gantry_publish_alpha=args.gantry_publish_alpha,
                publish_median=args.publish_median,
                gantry_frame=args.gantry_frame,
                gantry_yaw_deg=args.gantry_yaw_deg,
                gantry_pitch_deg=args.gantry_pitch_deg,
                gantry_roll_deg=args.gantry_roll_deg,
                gantry_sign=(
                    args.gantry_sign_x,
                    args.gantry_sign_y,
                    args.gantry_sign_z,
                ),
                ros_publish_hz=args.ros_publish_hz,
                sync_adaptive=args.sync_adaptive,
            )

            def _reset_vision_clock():
                session_origin.reset()
                session_time.reset()
                interpolator.history.clear()
                interpolator.last_real_time = None
                pos_smoother.reset()
                world_smoother.reset()
                ros_pub._cam_smoother.reset()
                ros_pub._gantry_smoother.reset()
                ros_pub._prev_gantry_center = None
            ros_pub.set_motion_start_callback(_reset_vision_clock)
        except ImportError as exc:
            print(f'[ROS] Disabled: rclpy not available ({exc})')

    encoder = topdown = None
    td_w = td_h = args.topdown_size
    scale_x = scale_y = 1.0
    stream_server = None
    use_http = False
    if not args.no_stream:
        stream_server = start_stream_server(args.stream_port)
        use_http = stream_server is not None

    display_ok = use_http
    display_branch_ok = display_ok and not args.stream_detect_only
    ros_jpeg_ok = ros_pub is not None and not display_ok
    camera_jpeg_q = queue.Queue(maxsize=2) if ros_pub is not None else None
    ros_jpeg_encoder = None
    camera_jpeg_min_period = (
        1.0 / max(1.0, float(args.camera_jpeg_fps)) if ros_jpeg_ok else None
    )
    last_ros_jpeg_mono = 0.0
    if display_ok:
        encoder = StreamEncoder(
            quality=args.jpeg_quality,
            jpeg_queue=camera_jpeg_q,
            use_http=use_http,
        )
        encoder.start()
        topdown = TopDownView(
            width=td_w, height=td_h,
            x_range=(-args.x_range, args.x_range),
            z_range=(args.z_min, args.z_max),
        )
        scale_x = args.display_width / args.width
        scale_y = args.display_height / args.height
    elif ros_jpeg_ok:
        ros_jpeg_encoder = StreamEncoder(
            quality=args.jpeg_quality,
            jpeg_queue=camera_jpeg_q,
            use_http=False,
        )
        ros_jpeg_encoder.start()
        scale_x = scale_y = 1.0

    if ros_pub and args.wait_motion:
        print('[MAIN] Vision t=0 starts at /gantry/enable (MOTION_START on /traj_cmd)')
    else:
        print(f'\n[MAIN] Session origin: X=0 Z=0 and t=0 at first frame of each run')
    print(f'[MAIN] Vision CSVs: vision_osc_{session_ts}.csv, vision_tags_{session_ts}.csv')
    print(f'[MAIN] Logger CSVs (if running): logger_sync_pose_*, logger_traj_cmd_*')
    print(f'[MAIN] Detection: {args.width}x{args.height} @ {args.fps} FPS')
    if display_ok:
        if args.stream_detect_only:
            print(f'[MAIN] Display: detection-frame stream '
                  f'{args.display_width}x{args.display_height} '
                  f'@ ≤{args.display_fps} FPS')
        else:
            print(f'[MAIN] Display: {args.display_width}x{args.display_height} '
                  f'@ {args.display_fps} FPS  lead={args.display_lead}')
        if use_http:
            host = _local_lan_ip()
            print(f'[MAIN] Web stream: http://{host}:{args.stream_port}/')
        if ros_pub is not None:
            print('[MAIN] ROS camera → /payload/camera/compressed (HD path)')
    elif ros_jpeg_ok:
        print(
            f'[MAIN] ROS camera → /payload/camera/compressed '
            f'@ {args.camera_jpeg_fps:.0f} Hz (detect resolution, no HD branch)')
    if ros_pub:
        print(f'[MAIN] ROS2 frame_id={args.frame_id}')
    print()

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        detect_out = cam.requestOutput(
            (args.width, args.height), dai.ImgFrame.Type.GRAY8, fps=args.fps,
        )
        detect_q = detect_out.createOutputQueue()
        detect_q.setMaxSize(1)
        detect_q.setBlocking(False)

        display_q = None
        if display_branch_ok:
            display_out = cam.requestOutput(
                (args.display_width, args.display_height),
                dai.ImgFrame.Type.GRAY8, fps=args.display_fps,
            )
            display_q = display_out.createOutputQueue()
            display_q.setMaxSize(1)
            display_q.setBlocking(False)

        pipeline.start()
        print('[MAIN] Running. Ctrl+C to stop.\n')

        device = pipeline.getDefaultDevice()
        calib = device.readCalibration()
        cam_matrix = np.array(
            calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_B, args.width, args.height,
            ), dtype=np.float64,
        )
        dist_coeffs = np.array(
            calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B),
            dtype=np.float64,
        )

        detect_fps_counter = 0
        stream_fps_counter = 0
        fps_time = time.monotonic()
        detect_fps = 0.0
        stream_fps = 0.0
        frame_count = miss_count = interp_count = 0
        latest_result = None
        display_X = 0.0
        display_Z = 0.0
        latest_X = latest_Y = latest_Z = None
        latest_interpolated = display_held = False
        last_good_detect_ts = None
        osc_result = None
        latest_per_tag = []
        latest_per_tag_display = []
        latest_detect_ts = None
        ros_pending = None
        last_display_frame_mono = 0.0
        last_detect_stream_mono = 0.0
        detect_stream_min_period = 1.0 / max(1.0, float(args.display_fps))

        while pipeline.isRunning():
            got_work = False
            ros_pending = None

            # Process every pending detect frame (not just the newest dropped one).
            while True:
                gray_msg = detect_q.tryGet()
                if gray_msg is None:
                    break
                got_work = True
                gray = gray_msg.getCvFrame()
                gray_ts = gray_msg.getTimestamp().total_seconds()
                run_ts = session_time.to_session(gray_ts)
                latest_detect_ts = run_ts
                now = time.monotonic()

                detect_fps_counter += 1
                frame_count += 1

                result = tracker.detect(gray, cam_matrix, dist_coeffs, gray_ts)
                interpolated = False
                X = Y = Z = None

                if result:
                    per_tag = result.get('per_tag', [])
                    tag_states = []
                    if per_tag:
                        tag_logger.log(run_ts, per_tag, session_origin)
                        tag_states = _session_per_tag(
                            per_tag, session_origin, session_time, gray_ts,
                        )
                        latest_per_tag = tag_states
                        latest_per_tag_display = tag_states
                    if 'X' in result:
                        X, Y, Z = result['X'], result['Y'], result['Z']
                        X, Y, Z = world_smoother.filter(X, Y, Z)
                        rcx, rcy, rcz = session_origin.to_relative_xyz(X, Y, Z)
                        rx, rz = rcx, rcz
                        interpolator.add_real(gray_ts, rx, rcy, rz)
                        osc_result = osc.update(
                            rx, rcy, rz, run_ts, detect_fps, interpolated=False,
                        )
                        latest_result = result
                        display_held = False
                        last_good_detect_ts = gray_ts
                        display_X, display_Z = pos_smoother.update(rx, rz)
                        if ros_pub:
                            ros_pending = {
                                'detect_ts': run_ts,
                                'tag_states': tag_states or None,
                                'primary_face': result.get('primary_face'),
                                'face_groups': tracker.face_groups,
                                'valid': True,
                                'interpolated': False,
                                'center_cam': (rcx, rcy, rcz),
                            }
                    else:
                        X = Y = Z = None
                        if ros_pub and tag_states:
                            ros_pending = {
                                'detect_ts': run_ts,
                                'tag_states': tag_states,
                                'primary_face': result.get('primary_face'),
                                'face_groups': tracker.face_groups,
                                'valid': True,
                                'interpolated': False,
                                'center_cam': None,
                            }
                else:
                    miss_count += 1
                    pred = interpolator.predict(gray_ts)
                    if pred is not None:
                        X, Y, Z = pred
                        interpolated = True
                        interp_count += 1
                        osc_result = osc.update(
                            X, Y, Z, run_ts, detect_fps, interpolated=True,
                        )
                        display_X, display_Z = pos_smoother.update(X, Z)
                        if ros_pub:
                            ros_pending = {
                                'detect_ts': run_ts,
                                'valid': True,
                                'interpolated': True,
                                'center_cam': None,
                            }
                    display_held = (
                        latest_result is not None
                        and last_good_detect_ts is not None
                        and (gray_ts - last_good_detect_ts) <= args.hold_sec
                    )
                    if ros_pub and not result and pred is None:
                        ros_pending = {
                            'detect_ts': run_ts,
                            'valid': False,
                            'interpolated': False,
                            'center_cam': None,
                        }

                latest_X, latest_Y, latest_Z = X, Y, Z
                latest_interpolated = interpolated

                if ros_jpeg_encoder is not None and camera_jpeg_min_period is not None:
                    if now - last_ros_jpeg_mono >= camera_jpeg_min_period:
                        last_ros_jpeg_mono = now
                        view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        if latest_result is not None and 'face_detections' in latest_result:
                            tags_draw = tracker.tags_for_overlay(
                                latest_result.get('primary_face'),
                                latest_result['face_detections'],
                                gray_ts, args.hold_sec,
                            )
                            _draw_overlay(
                                view, tags_draw, scale_x, scale_y, 0.0, display_held,
                            )
                            if latest_per_tag_display:
                                _draw_per_tag_labels(
                                    view, tags_draw, latest_per_tag_display,
                                    scale_x, scale_y, 0.0, display_held,
                                )
                        cv2.putText(
                            view,
                            f'det {detect_fps:.0f}',
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                        )
                        ros_jpeg_encoder.submit(view)

                if (
                    encoder is not None
                    and (
                        args.stream_detect_only
                        or (display_q is not None and (now - last_display_frame_mono) > 0.5)
                    )
                    and (now - last_detect_stream_mono) >= detect_stream_min_period
                ):
                    last_detect_stream_mono = now
                    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    if latest_result is not None and 'face_detections' in latest_result:
                        tags_draw = tracker.tags_for_overlay(
                            latest_result.get('primary_face'),
                            latest_result['face_detections'],
                            gray_ts, args.hold_sec,
                        )
                        _draw_overlay(
                            view, tags_draw, 1.0, 1.0, 0.0, display_held,
                        )
                        if latest_per_tag_display:
                            _draw_per_tag_labels(
                                view, tags_draw, latest_per_tag_display,
                                1.0, 1.0, 0.0, display_held,
                            )
                    cv2.putText(
                        view,
                    f'det {detect_fps:.0f}  stream detect',
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    )
                    view = cv2.resize(
                        view, (args.display_width, args.display_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    if topdown is not None:
                        td = topdown.render(
                            (display_X, display_Z),
                            interpolated=latest_interpolated or display_held,
                            update_trail=not display_held and not latest_interpolated,
                        )
                        if td.shape[0] != args.display_height:
                            td = cv2.resize(
                                td, (args.display_height, args.display_height))
                        encoder.submit(np.hstack([view, td]))
                    else:
                        encoder.submit(view)

            if ros_pub:
                if camera_jpeg_q is not None:
                    while True:
                        try:
                            jpeg = camera_jpeg_q.get_nowait()
                        except queue.Empty:
                            break
                        ros_pub.publish_camera(jpeg)
                if ros_pending is not None:
                    ros_pub.update(**ros_pending)
                ros_pub.spin_once()

            tick = time.monotonic()
            if tick - fps_time >= 1.0:
                elapsed = tick - fps_time
                detect_fps = detect_fps_counter / elapsed
                stream_fps = stream_fps_counter / elapsed
                detect_fps_counter = 0
                stream_fps_counter = 0
                fps_time = tick
                if ros_pub is not None:
                    ros_pub.note_detect_fps(detect_fps)
                if frame_count % 120 == 0 and latest_X is not None and osc_result:
                    miss_pct = 100.0 * miss_count / max(1, frame_count)
                    print(
                        f'[detect {detect_fps:.0f} FPS | stream {stream_fps:.0f} FPS] '
                        f'X={latest_X:+.4f} Z={latest_Z:.4f} '
                        f'miss={miss_pct:.1f}%'
                    )
                    if latest_detect_ts is not None:
                        for t in latest_per_tag:
                            print(
                                f'  tag {t["tag_id"]}: '
                                f't={t["time"]:.3f}s '
                                f'x={t["x"]:+.4f}m z={t["z"]:.4f}m '
                                f'dx/dt={t["dx_dt"]:+.4f} m/s'
                            )

            if encoder and display_q is not None:
                hd_msg = _drain_queue(display_q)
                if hd_msg is not None:
                    got_work = True
                    stream_fps_counter += 1
                    last_display_frame_mono = time.monotonic()

                    hd_gray = hd_msg.getCvFrame()
                    hd_view = cv2.cvtColor(hd_gray, cv2.COLOR_GRAY2BGR)
                    hd_ts = hd_msg.getTimestamp().total_seconds()

                    if latest_result is not None and 'face_detections' in latest_result:
                        dt_display = _display_dt(
                            hd_ts, latest_result['detect_timestamp'],
                            display_held, args.hold_sec, args.display_lead,
                        )
                        tags_draw = tracker.tags_for_overlay(
                            latest_result.get('primary_face'),
                            latest_result['face_detections'],
                            hd_ts, args.hold_sec,
                        )
                        _draw_overlay(
                            hd_view, tags_draw, scale_x, scale_y,
                            dt_display, display_held,
                        )
                        if latest_per_tag_display:
                            _draw_per_tag_labels(
                                hd_view, tags_draw, latest_per_tag_display,
                                scale_x, scale_y, dt_display, display_held,
                            )
                    elif latest_per_tag_display:
                        _draw_per_tag_labels(
                            hd_view, [], latest_per_tag_display,
                            scale_x, scale_y, 0.0, display_held,
                        )

                    cv2.putText(
                        hd_view,
                        f'det {detect_fps:.0f}  stream {stream_fps:.0f}',
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    )
                    freq_s = (
                        f'  f={osc_result["freq_hz"]:.2f}Hz' if osc_result else ''
                    )
                    cv2.putText(
                        hd_view,
                        f'X={display_X:+.3f}m  Z={display_Z:.3f}m{freq_s}',
                        (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                    if display_held or latest_interpolated:
                        cv2.putText(
                            hd_view, 'HOLD' if display_held else 'INTERP',
                            (args.display_width - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                        )

                    if topdown is not None:
                        marker_xz = (display_X, display_Z)
                        td = topdown.render(
                            marker_xz,
                            interpolated=latest_interpolated or display_held,
                            update_trail=not display_held and not latest_interpolated,
                        )
                        if td.shape[0] != args.display_height:
                            td = cv2.resize(
                                td, (args.display_height, args.display_height))
                        encoder.submit(np.hstack([hd_view, td]))
                    else:
                        encoder.submit(hd_view)

            if not got_work:
                time.sleep(0.001)

    osc.close()
    tag_logger.close()
    if ros_pub:
        ros_pub.shutdown()
    print('[MAIN] Done.')


if __name__ == '__main__':
    main()
