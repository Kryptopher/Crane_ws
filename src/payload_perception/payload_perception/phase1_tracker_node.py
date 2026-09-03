#!/usr/bin/env python3
"""
ROS2 node wrapping the Phase 1 cube-face AprilTag tracker.

Publishes PayloadState on /payload/state with motion_time_sec synced to the
goto move_done falling edge from /gantry/state (t=0 = cart starts moving).

Camera-only: no encoder data is mixed into the published state.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Make the standalone tracker importable.
_OPS_DIR = os.path.join(os.path.expanduser("~"), "crane_ws", "ops", "Payload_Tracker_Updated")
if _OPS_DIR not in sys.path:
    sys.path.insert(0, _OPS_DIR)

from phase1_tracker import (
    CubeEstimator,
    CubePoseResult,
    FaceLayout,
    MotionTracker,
    PoseResult,
    SingleFacePoseEstimator,
    _MJPEGStream,
    _WebData,
    _start_stream_server,
    _stream,
    _webdata,
    draw_tracking,
    load_config,
    _local_ip,
    _CONV_MSG,
)

import cv2
import depthai as dai

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from payload_perception_msgs.msg import PayloadState
from gantry_control.msg import GantryState
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, Float64MultiArray, Empty, String


_SYNC_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    depth=1,
)


class Phase1TrackerNode(Node):
    def __init__(self, cfg: dict):
        super().__init__("payload_tracker")
        self._nan = float("nan")

        # ── Config ──
        cam_cfg = cfg.get("camera", {})
        disp_cfg = cfg.get("display", {})
        self._stream_port = int(disp_cfg.get("stream_port", 8092))
        self._jpeg_quality = int(disp_cfg.get("jpeg_quality", 72))
        self._stream_enabled = bool(disp_cfg.get("enable_stream", True))

        # Camera swing correction, from fit_camera_encoder_calibration.py.
        # Applied as swing_mm_corrected = swing_mm * scale + offset_mm at
        # every point camera swing is computed (live state + experiment log)
        # so control, dashboard, and logged CSVs all see the same value.
        #
        # The camera is a fixed tripod camera (not mounted on the cart), so
        # its solvePnP-recovered scale can vary with the cart's absolute
        # position along the rail — a single scale value is only exactly
        # right at whatever position it was fit at. If face_config.yaml has a
        # `scale_vs_x: {m, c}` block (from fit_camera_encoder_calibration.py's
        # position-dependence check, run via run_calibration_sweep.py
        # --positions-mm), evaluate the scale at the cart's live position
        # instead of using one flat constant.
        cal_cfg = cfg.get("calibration", {})
        self._camera_swing_scale = float(cal_cfg.get("camera_swing_scale", 1.0))
        self._camera_swing_offset_mm = float(cal_cfg.get("camera_swing_offset_mm", 0.0))
        self._camera_swing_scale_vs_x = cal_cfg.get("scale_vs_x")

        # ── Tracker core (multi-face: all instrumented cube faces) ──
        self._estimator = CubeEstimator(cfg)
        self._layout = self._estimator.layout   # primary face, for drawing
        self._motion = MotionTracker(
            gap_s=float(cfg.get("motion", {}).get("gap_s", 0.25)),
        )

        # ── Gantry + encoder state tracking ──
        self._prev_move_done: bool = True
        self._lock = threading.Lock()
        self._cart_x: float = 0.0
        self._cart_y: float = 0.0
        self._cart_vx: float = 0.0
        self._swing_x: float = 0.0
        self._swing_vx: float = 0.0
        self._swing_y: float = 0.0
        self._swing_vy: float = 0.0
        self._start_x_m: float = 0.350
        # Gantry X/Y (abs) captured when calibration is applied.
        self._gantry_x_at_cal: float = self._start_x_m
        self._gantry_y_at_cal: float = 0.0
        # Cart-only and swing baselines at calibration — experiment logs record
        # SWING (payload relative to cart): camera_disp − cart_disp vs rope
        # encoder x_rel, both re-based to the run start.
        self._cart_x_at_cal: float = 0.0
        self._swing_x_at_cal: float = 0.0
        # Calibration is applied ATOMICALLY in the camera thread on the next
        # frame with a valid pose: gantry anchor snapshot + motion origin reset
        # happen together.  Applying immediately on /payload/reset_origin
        # half-applied when tags weren't tracked (anchor moved, camera
        # displacement origin didn't) — the (400,700) constant-offset bug.
        self._cal_pending: bool = False

        # ── Tracking error stats (gantry X vs camera X, gantry Y vs camera depth) ──
        self._cam_disp_mm: float = 0.0          # latest camera X displacement
        self._cam_depth_mm: float = 0.0         # latest camera depth (Z) displacement
        self._error_now_mm: float = 0.0         # current X (gantry - camera) error
        self._depth_error_now_mm: float = 0.0   # current depth (gantry Y - camera Z) error
        self._error_max_mm: float = 0.0         # peak |error| during last movement
        self._error_final_mm: Optional[float] = None  # X error when last move completed
        self._depth_error_final_mm: Optional[float] = None

        # ── ROS publishers ──
        self.pub_state = self.create_publisher(
            PayloadState, "/payload/state", qos_profile_sensor_data)
        self.pub_camera = self.create_publisher(
            CompressedImage, "/payload/camera/compressed", qos_profile_sensor_data)
        self.pub_sync_hz = self.create_publisher(
            Float64, "/stack/pose_sync_hz", _SYNC_QOS)
        self._camera_pub_interval = 1.0 / 15.0  # 15 fps to dashboard
        self._camera_pub_last = 0.0
        self.create_subscription(
            GantryState, "/gantry/state", self._on_gantry_state, 10)
        self.create_subscription(
            Float64MultiArray, "/payload/pose_e_rel",
            self._on_encoder_rel, qos_profile_sensor_data)
        self.create_subscription(
            Empty, "/payload/reset_origin", self._on_reset_origin, 10)
        self.create_subscription(
            Empty, "/experiment/start", self._on_experiment_start, 10)
        self.create_subscription(
            Empty, "/experiment/end", self._on_experiment_end, 10)
        # Commanded-run flow: stop holds the recorded data (no save) so the
        # operator can title it; save writes plot+CSVs with that title.
        # /experiment/end keeps its auto-save behaviour for the legacy scripts.
        self.create_subscription(
            Empty, "/experiment/stop", self._on_experiment_stop, 10)
        self.create_subscription(
            String, "/experiment/save", self._on_experiment_save, 10)
        self.create_subscription(
            Float64, "/experiment/set_start_x", self._on_set_start_x, 10)

        # ── Sync Hz tracking ──
        self._frame_count: int = 0
        self._hz_window_start: float = time.monotonic()
        self._sync_hz_timer = self.create_timer(0.5, self._publish_sync_hz)

        # ── Experiment mode ──
        self._experiment_active: bool = False
        self._experiment_t0: Optional[float] = None
        self._experiment_log: list[tuple[float, float, float]] = []
        self._experiment_gantry_log: list[tuple[float, float, float]] = []

        # ── Web stream ──
        if self._stream_enabled:
            _start_stream_server(self._stream_port)
            self.get_logger().info(
                f"Stream: http://{_local_ip()}:{self._stream_port}/")
        else:
            self.get_logger().info(
                "Stream disabled (display.enable_stream: false in face_config.yaml)")

        # ── Camera pipeline (runs in a thread) ──
        self._oak_ip = cam_cfg.get("ip", "") or os.environ.get("OAK_IP", "")
        self._socket_name = str(cam_cfg.get("socket", "CAM_B"))
        self._width = int(cam_cfg.get("detect_width", 640))
        self._height = int(cam_cfg.get("detect_height", 400))
        self._fps = int(cam_cfg.get("detect_fps", 60))
        self._axis_len = float(disp_cfg.get("axis_length_m", 0.05))

        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

        self.get_logger().info("Phase 1 tracker node started. Waiting for goto move...")

    # ── Time sync: move_done falling edge = t=0 ──

    def _on_gantry_state(self, msg: GantryState) -> None:
        prev_done = self._prev_move_done
        new_done = bool(msg.move_done)
        self._prev_move_done = new_done
        with self._lock:
            self._cart_x = msg.x
            self._cart_y = msg.y
            self._cart_vx = msg.vx
        self._update_encoder_webdata()
        if prev_done and not new_done:          # movement started — reset peak
            self._error_max_mm = 0.0
        if not prev_done and new_done:          # movement ended — log final error
            self._error_final_mm = self._error_now_mm
            self._depth_error_final_mm = self._depth_error_now_mm
            self.get_logger().info(
                f"Move done  X err: final={self._error_final_mm:+.1f} mm  "
                f"peak={self._error_max_mm:.1f} mm  "
                f"Z err: {self._depth_error_final_mm:+.1f} mm"
            )

    def _on_encoder_rel(self, msg: Float64MultiArray) -> None:
        d = msg.data
        if len(d) < 9:
            return
        with self._lock:
            self._swing_x = d[3]   # x_rel_m
            self._swing_y = d[4]   # y_rel_m
            self._swing_vx = d[6]  # vx_rel_m_s
            self._swing_vy = d[7]  # vy_rel_m_s

    def _update_encoder_webdata(self) -> None:
        with self._lock:
            abs_x = self._cart_x + self._swing_x
            abs_vx = self._cart_vx + self._swing_vx
            cart_y = self._cart_y
            cam_disp = self._cam_disp_mm
            cam_depth = self._cam_depth_mm
        disp_mm = (abs_x - self._gantry_x_at_cal) * 1000.0
        depth_disp_mm = (cart_y - self._gantry_y_at_cal) * 1000.0
        vx_mm_s = abs_vx * 1000.0

        err_mm = disp_mm - cam_disp
        depth_err_mm = depth_disp_mm - cam_depth
        self._error_now_mm = err_mm
        self._depth_error_now_mm = depth_err_mm
        if not self._prev_move_done:            # track peak only during active movement
            if abs(err_mm) > self._error_max_mm:
                self._error_max_mm = abs(err_mm)

        _webdata.set(
            enc_vel=round(vx_mm_s, 1),
            enc_disp=round(disp_mm, 1),
            enc_active=True,
            depth_disp=round(depth_disp_mm, 1),
            depth_err=round(depth_err_mm, 1),
        )

        if self._experiment_active and self._experiment_t0 is not None:
            exp_t = time.monotonic() - self._experiment_t0
            # Rope-encoder payload swing relative to the cart, re-based to the
            # swing at experiment start (matches the camera swing trace).
            with self._lock:
                swing_mm = (self._swing_x - self._swing_x_at_cal) * 1000.0
                swing_vx_mm_s = self._swing_vx * 1000.0
            self._experiment_gantry_log.append((exp_t, swing_mm, swing_vx_mm_s))
            if not _webdata._d.get("first_responder"):
                with _webdata._lock:
                    if not _webdata._d["first_responder"]:
                        _webdata._d["first_responder"] = "encoder"

    def _on_set_start_x(self, msg: Float64) -> None:
        self._start_x_m = msg.data / 1000.0
        self.get_logger().info(f"Start X offset set to {msg.data:.1f} mm")

    def _on_reset_origin(self, _msg: Empty) -> None:
        self._cal_pending = True
        self.get_logger().info(
            "/payload/reset_origin — calibration queued, applies on next tracked frame")

    def _current_camera_swing_scale(self) -> float:
        if self._camera_swing_scale_vs_x is None:
            return self._camera_swing_scale
        with self._lock:
            cart_x_mm = self._cart_x * 1000.0
        model = self._camera_swing_scale_vs_x
        return float(model["m"]) * cart_x_mm + float(model["c"])

    def _apply_calibration(self) -> None:
        """Snapshot the gantry anchor and re-zero the camera displacement
        origin in one step.  Called from the camera thread immediately after a
        successful motion update, so the origin lands on the current pose."""
        self._motion.reset_origin()
        with self._lock:
            # Payload absolute position = cart + rope swing, on BOTH axes
            # (Y previously ignored swing — anchor asymmetry).
            self._gantry_x_at_cal = self._cart_x + self._swing_x
            self._gantry_y_at_cal = self._cart_y + self._swing_y
            self._cart_x_at_cal = self._cart_x
            self._swing_x_at_cal = self._swing_x
            self._cam_disp_mm = 0.0
            self._cam_depth_mm = 0.0
            self._cal_pending = False
        # Restart the overlay error stats at the new reference.
        self._error_max_mm = 0.0
        self._error_final_mm = None
        self._depth_error_final_mm = None
        self.get_logger().info(
            f"Calibration applied — X_cal={self._gantry_x_at_cal*1000:.0f}mm "
            f"Y_cal={self._gantry_y_at_cal*1000:.0f}mm"
        )

    def _on_experiment_start(self, _msg: Empty) -> None:
        if self._experiment_active:
            self.get_logger().warn("Experiment already active — ignoring start")
            return
        self._experiment_log.clear()
        self._experiment_gantry_log.clear()
        # Atomic re-zero (anchor + origin together) on the next tracked frame.
        self._cal_pending = True
        self._experiment_t0 = time.monotonic()
        self._experiment_active = True
        _webdata.set(experiment="recording", first_responder="")
        self.get_logger().info("Experiment started — t=0")

    def _on_experiment_end(self, _msg: Empty) -> None:
        if not self._experiment_active:
            self.get_logger().warn("No experiment active — ignoring end")
            return
        self._experiment_active = False
        n_cam = len(self._experiment_log)
        n_gantry = len(self._experiment_gantry_log)
        self.get_logger().info(
            f"Experiment ended — {n_cam} camera + {n_gantry} gantry samples")
        _webdata.set(experiment="saved")
        self._save_experiment()

    def _on_experiment_stop(self, _msg: Empty) -> None:
        """Stop recording but hold the data until /experiment/save arrives."""
        if not self._experiment_active:
            self.get_logger().warn("No experiment active — ignoring stop")
            return
        self._experiment_active = False
        _webdata.set(experiment="stopped")
        self.get_logger().info(
            f"Experiment stopped — {len(self._experiment_log)} camera + "
            f"{len(self._experiment_gantry_log)} gantry samples held for save")

    def _on_experiment_save(self, msg: String) -> None:
        if self._experiment_active:
            self._experiment_active = False
        if len(self._experiment_log) < 2:
            self.get_logger().warn("No experiment data held — ignoring save")
            return
        title = msg.data.strip() or None
        _webdata.set(experiment="saved")
        self.get_logger().info(f"Saving experiment (title={title!r})")
        self._save_experiment(title)

    # ── Publish sync Hz ──

    def _publish_sync_hz(self) -> None:
        now = time.monotonic()
        dt = now - self._hz_window_start
        if dt > 0.1 and self._frame_count > 0:
            hz = self._frame_count / dt
            msg = Float64()
            msg.data = hz
            self.pub_sync_hz.publish(msg)
        self._frame_count = 0
        self._hz_window_start = now

    # ── PayloadState publish ──

    def _publish_pose(self, pose: PoseResult, motion_t: float,
                      disp_x_mm: float, disp_z_mm: float) -> None:
        nan = self._nan
        state = PayloadState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.header.frame_id = "camera"
        state.motion_time_sec = motion_t

        # Legacy tag-pair fields: not applicable to single-face solver
        state.x1 = nan
        state.z1 = nan
        state.x2 = nan
        state.z2 = nan
        state.vx1 = nan
        state.vz1 = nan
        state.vx2 = nan
        state.vz2 = nan

        # Cube centre in camera frame (fused across visible faces) — the same
        # physical point regardless of which face is tracked.
        tv = pose.cube_tvec
        state.cam_x = float(tv[0])
        state.cam_y = float(tv[1])
        state.cam_z = float(tv[2])

        # Camera X is sign-flipped to match gantry convention (gantry left = +X,
        # camera left = -X).  gantry_x anchors camera displacement to the gantry
        # absolute position captured at reset_origin.
        state.gantry_x = self._gantry_x_at_cal + disp_x_mm / 1000.0
        state.gantry_y = self._gantry_y_at_cal + disp_z_mm / 1000.0
        state.gantry_z = nan
        state.v_gantry_x = nan
        state.v_gantry_y = nan
        state.v_gantry_z = nan

        state.valid = True
        state.interpolated = False
        # Surface which face anchors the solve (shows in the dashboard meta line).
        state.frame_id = (f"camera:{pose.face_name}"
                          if isinstance(pose, CubePoseResult) else "camera")

        self.pub_state.publish(state)

    # ── Camera loop (blocking, runs in thread) ──

    def _camera_loop(self) -> None:
        if self._oak_ip:
            os.environ["DEPTHAI_DEVICE_NAME"] = self._oak_ip

        from phase1_tracker import _camera_socket
        socket_id = _camera_socket(self._socket_name)

        telemetry_interval = 1.0
        prev_tel = time.monotonic()

        with dai.Pipeline() as pipeline:
            cam_node = pipeline.create(dai.node.Camera).build(socket_id)
            out_node = cam_node.requestOutput(
                (self._width, self._height), dai.ImgFrame.Type.GRAY8, fps=self._fps
            )
            q = out_node.createOutputQueue()
            q.setMaxSize(1)
            q.setBlocking(False)

            pipeline.start()
            device = pipeline.getDefaultDevice()
            calib = device.readCalibration()
            K = np.array(
                calib.getCameraIntrinsics(socket_id, self._width, self._height),
                dtype=np.float64,
            )
            dist = np.array(
                calib.getDistortionCoefficients(socket_id), dtype=np.float64
            ).reshape(-1)

            self.get_logger().info(
                f"Camera {self._socket_name} {self._width}x{self._height} @ {self._fps} fps"
            )

            while rclpy.ok():
                msg = q.tryGet()
                if msg is None:
                    time.sleep(0.001)
                    continue

                gray = msg.getCvFrame()
                undist = self._estimator.undistort(gray, K, dist)
                dets = self._estimator.detect_face(undist)
                pose = self._estimator.estimate_pose(dets, K)

                frame = draw_tracking(
                    undist, dets, pose, K,
                    layout=self._layout,
                    axis_len=self._axis_len,
                    debug_corners=False,
                )

                # ── Error overlay (bottom of frame) ──
                fh = frame.shape[0]
                _font = cv2.FONT_HERSHEY_SIMPLEX
                _err = self._error_now_mm
                _derr = self._depth_error_now_mm
                def _err_clr(e):
                    return ((0, 220, 0) if abs(e) < 5.0
                            else (0, 165, 255) if abs(e) < 20.0
                            else (0, 0, 255))
                _lbl_live = f"X err: {_err:+.1f}mm   Z err: {_derr:+.1f}mm"
                cv2.putText(frame, _lbl_live, (12, fh - 36), _font, 0.50, (0, 0, 0), 3)
                cv2.putText(frame, _lbl_live, (12, fh - 36), _font, 0.50, _err_clr(_err), 1)
                if self._error_final_mm is not None:
                    _lbl = (f"last move  X final={self._error_final_mm:+.1f}mm  "
                            f"peak={self._error_max_mm:.1f}mm")
                    cv2.putText(frame, _lbl, (12, fh - 12), _font, 0.45, (0, 0, 0), 3)
                    cv2.putText(frame, _lbl, (12, fh - 12), _font, 0.45, (200, 200, 200), 1)

                if self._stream_enabled:
                    _stream.update(frame, quality=self._jpeg_quality)

                    now_pub = time.monotonic()
                    if (now_pub - self._camera_pub_last) >= self._camera_pub_interval:
                        self._camera_pub_last = now_pub
                        ok, buf = cv2.imencode(".jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                        if ok:
                            img_msg = CompressedImage()
                            img_msg.header.stamp = self.get_clock().now().to_msg()
                            img_msg.format = "jpeg"
                            img_msg.data = buf.tobytes()
                            self.pub_camera.publish(img_msg)

                self._frame_count += 1

                if _webdata.take_reset():
                    # Web-page reset button: route through the same atomic
                    # calibration so anchor and origin never diverge.
                    self._cal_pending = True

                now = time.monotonic()

                if pose is not None:
                    # Track the fused CUBE CENTRE (face-invariant): displacement
                    # stays continuous when the tracked face changes, unlike a
                    # face-centre origin which jumps ~50-70 mm on face handoff.
                    self._motion.update(pose.cube_tvec, now)
                    if self._cal_pending:
                        self._apply_calibration()
                    with self._lock:
                        cached_disp_x = self._motion.displacement_x_mm
                        cached_disp_z = self._motion.displacement_z_mm
                        self._cam_disp_mm  = cached_disp_x
                        self._cam_depth_mm = cached_disp_z
                    self._publish_pose(pose,
                        (now - self._experiment_t0) if self._experiment_t0 else 0.0,
                        cached_disp_x, cached_disp_z)

                    if self._experiment_active and self._experiment_t0 is not None:
                        exp_t = now - self._experiment_t0
                        # Camera-measured payload SWING: cube-centre displacement
                        # minus the cart's own displacement (motor encoders), so
                        # it compares directly against the rope encoder's x_rel.
                        with self._lock:
                            cart_dx_mm = (self._cart_x - self._cart_x_at_cal) * 1000.0
                            cart_vx_mm_s = self._cart_vx * 1000.0
                        # camera_swing_scale/offset_mm come from
                        # fit_camera_encoder_calibration.py, fit against this
                        # exact swing quantity (see calibration: block in
                        # face_config.yaml). Applied only here, not to the
                        # absolute cam_disp webdata field above, since that
                        # feeds a separate camera-vs-encoder absolute-position
                        # accuracy check this correction was never fit against.
                        raw_swing_mm = self._motion.displacement_x_mm - cart_dx_mm
                        raw_swing_vx_mm_s = self._motion.vx_mm_s - cart_vx_mm_s
                        swing_scale = self._current_camera_swing_scale()
                        self._experiment_log.append((
                            exp_t,
                            raw_swing_mm * swing_scale
                            + self._camera_swing_offset_mm,
                            raw_swing_vx_mm_s * swing_scale,
                        ))
                        if not _webdata._d.get("first_responder"):
                            with _webdata._lock:
                                if not _webdata._d["first_responder"]:
                                    _webdata._d["first_responder"] = "camera"

                    _webdata.set(
                        tracking=True,
                        ids=pose.visible_ids,
                        cam_vel=round(self._motion.vx_mm_s, 1),
                        cam_disp=round(self._motion.displacement_x_mm, 1),
                    )

                    if now - prev_tel >= telemetry_interval:
                        prev_tel = now
                        c = pose.cube_tvec * 1000.0
                        self.get_logger().info(
                            f"cube=[{c[0]:+.1f} {c[1]:+.1f} {c[2]:+.1f}]mm "
                            f"vx={self._motion.vx_mm_s:+.1f}mm/s "
                            f"disp_x={self._motion.displacement_x_mm:+.1f}mm "
                            f"reproj={pose.reproj_px:.2f}px"
                        )
                else:
                    self._motion.on_lost()
                    _webdata.set(
                        tracking=False, ids=[],
                        cam_vel=0.0,
                    )

    # ── Save experiment ──

    def _save_experiment(self, title: Optional[str] = None) -> None:
        n = len(self._experiment_log)
        if n < 2:
            self.get_logger().warn(f"Only {n} camera samples — not enough to save.")
            return
        from phase1_tracker import _save_session_plot
        _save_session_plot(self._experiment_log, self._experiment_gantry_log,
                           title=title,
                           disp_name="Payload swing X (relative to cart)",
                           vel_name="Payload swing X velocity",
                           csv_col="swing_x_mm")
        self._experiment_log.clear()
        self._experiment_gantry_log.clear()


def main(args=None):
    rclpy.init(args=args)

    cfg_path = Path(_OPS_DIR) / "face_config.yaml"
    cfg = load_config(cfg_path if cfg_path.exists() else None)

    node = Phase1TrackerNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._experiment_active:
            node._experiment_active = False
            node._save_experiment()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
