"""
Axis conventions for payload_perception.

Camera (OAK optical, OpenCV-style):
  X    — horizontal in the image (right = +X; cart X+ often looks like -cam_x / “left”)
  Z    — depth along the view axis (cart Y+ → closer → cam_z changes)
  Y    — vertical in the image

Gantry (crane cart):
  X, Y — horizontal cart travel (CoreXY)
  Z    — out-of-plane payload swing (pendulum)

Tracker: static rotation only (tripod-fixed camera; no /gantry/state).
Cart compensation: payload_gantry_frame — only if camera rides on cart (not this lab).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    from gantry_control.msg import GantryState, TrajCmd
except ImportError:
    GantryState = None  # type: ignore
    TrajCmd = None  # type: ignore


def rotation_camera_to_gantry(
    yaw_deg: float, pitch_deg: float, roll_deg: float,
) -> np.ndarray:
    """Z(yaw) * Y(pitch) * X(roll): camera (x,y,z) → gantry (x,y,z)."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    cz, sz = math.cos(y), math.sin(y)
    cy, sy = math.cos(p), math.sin(p)
    cx, sx = math.cos(r), math.sin(r)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx


def camera_xyz_to_gantry(
    cam_x: float,
    cam_y: float,
    cam_z: float,
    R: np.ndarray,
    sign: np.ndarray,
) -> Tuple[float, float, float]:
    p = sign * np.array([cam_x, cam_y, cam_z], dtype=float)
    g = R @ p
    return float(g[0]), float(g[1]), float(g[2])


class CameraToGantryRotation:
    """Fixed mount: camera optical → gantry axes. No cart position required."""

    def __init__(
        self,
        *,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        sign_x: float = 1.0,
        sign_y: float = 1.0,
        sign_z: float = 1.0,
    ):
        self._R = rotation_camera_to_gantry(yaw_deg, pitch_deg, roll_deg)
        self._sign = np.array([sign_x, sign_y, sign_z], dtype=float)

    def to_gantry(
        self, cam_x: float, cam_y: float, cam_z: float,
    ) -> Tuple[float, float, float]:
        return camera_xyz_to_gantry(cam_x, cam_y, cam_z, self._R, self._sign)


class CartMotionCompensator:
    """
    Remove cart translation from gantry-frame payload pose.
    Requires /gantry/state (camera is on the moving cart).
    """

    def __init__(self, node, *, enabled: bool = True):
        if GantryState is None or TrajCmd is None:
            raise RuntimeError('gantry_control messages required')

        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self._enabled = enabled
        self._cart_x = 0.0
        self._cart_y = 0.0
        self._cart_x0: Optional[float] = None
        self._cart_y0: Optional[float] = None

        if not enabled:
            return

        traj_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        node.create_subscription(GantryState, '/gantry/state', self._on_gantry, 10)
        node.create_subscription(TrajCmd, '/traj_cmd', self._on_traj, traj_qos)
        node.get_logger().info('Cart motion compensation enabled (/gantry/state)')

    def _on_gantry(self, msg: 'GantryState'):
        self._cart_x = float(msg.x)
        self._cart_y = float(msg.y)

    def _on_traj(self, msg: 'TrajCmd'):
        if msg.command != TrajCmd.MOTION_START:
            return
        self._cart_x0 = self._cart_x
        self._cart_y0 = self._cart_y

    def reset_reference(self):
        self._cart_x0 = self._cart_x
        self._cart_y0 = self._cart_y

    def apply(
        self, gx: float, gy: float, gz: float,
    ) -> Tuple[float, float, float]:
        if (
            not self._enabled
            or self._cart_x0 is None
            or self._cart_y0 is None
        ):
            return gx, gy, gz
        return (
            gx - (self._cart_x - self._cart_x0),
            gy - (self._cart_y - self._cart_y0),
            gz,
        )
