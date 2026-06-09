#!/usr/bin/env python3
"""
payload_gantry_frame — Cart-motion compensation (camera ON the cart only).

NOT used when OAK is on a fixed tripod (this lab). Use payload_tracker gantry_* only.

Only enable if the camera is rigidly mounted on the moving gantry cart.

Subscribes:
  /payload/state   gantry_x/y/z (from tracker rotation)
  /gantry/state    cart x,y
  /traj_cmd        MOTION_START → reset cart reference

Publishes:
  /payload/pose_gantry   [motion_time_sec, x, y, z, vx, vy, vz]  (compensated, m)
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from payload_perception.payload_frames import CartMotionCompensator

try:
    from payload_perception_msgs.msg import PayloadState
except ImportError:
    PayloadState = None  # type: ignore


class PayloadGantryFrame(Node):
    def __init__(self):
        super().__init__('payload_gantry_frame')

        if PayloadState is None:
            raise RuntimeError('payload_perception_msgs required')

        self.declare_parameter('compensate_cart', True)
        compensate = bool(self.get_parameter('compensate_cart').value)

        self._cart = CartMotionCompensator(self, enabled=compensate)
        self._prev: Optional[tuple[float, float, float, float]] = None

        self.create_subscription(
            PayloadState, '/payload/state', self._on_state, qos_profile_sensor_data)
        self._pub = self.create_publisher(
            Float64MultiArray, '/payload/pose_gantry', qos_profile_sensor_data)

        self.get_logger().info(
            'Cart compensation on tracker gantry_x/y/z → /payload/pose_gantry')

    def _on_state(self, msg: 'PayloadState'):
        if not msg.valid:
            return
        if not (
            math.isfinite(msg.gantry_x)
            and math.isfinite(msg.gantry_y)
            and math.isfinite(msg.gantry_z)
        ):
            return

        gx, gy, gz = self._cart.apply(
            float(msg.gantry_x), float(msg.gantry_y), float(msg.gantry_z),
        )
        t = float(msg.motion_time_sec)

        vx = vy = vz = 0.0
        if self._prev is not None:
            t0, px, py, pz = self._prev
            dt = t - t0
            if dt > 1e-4:
                vx = (gx - px) / dt
                vy = (gy - py) / dt
                vz = (gz - pz) / dt
        self._prev = (t, gx, gy, gz)

        out = Float64MultiArray()
        dim = MultiArrayDimension()
        dim.label = 'motion_time_sec,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s'
        dim.size = 7
        dim.stride = 7
        out.layout.dim.append(dim)
        out.data = [t, gx, gy, gz, vx, vy, vz]
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PayloadGantryFrame()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
