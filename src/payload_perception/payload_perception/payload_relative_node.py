#!/usr/bin/env python3
"""
payload_relative_node.py

Computes payload displacement relative to the moving gantry/trolley.

Subscribes:
  /payload/state    payload_perception_msgs/PayloadState
  /gantry/state     gantry_control/GantryState

Publishes:
  /payload/relative_state  payload_perception_msgs/PayloadState

Meaning:
  relative.gantry_x = payload.gantry_x - gantry.x
  relative.gantry_y = payload.gantry_y - gantry.y
  relative.gantry_z = payload.gantry_z

This gives the adaptive input shaper the payload displacement relative to
the moving support.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from gantry_control.msg import GantryState
from payload_perception_msgs.msg import PayloadState


class PayloadRelativeNode(Node):
    def __init__(self):
        super().__init__("payload_relative_node")

        self.latest_gantry_state: GantryState | None = None

        self.create_subscription(
            GantryState,
            "/gantry/state",
            self._gantry_state_cb,
            10,
        )

        self.create_subscription(
            PayloadState,
            "/payload/state",
            self._payload_state_cb,
            qos_profile_sensor_data,
        )

        self.pub = self.create_publisher(
            PayloadState,
            "/payload/relative_state",
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "payload_relative_node started: /payload/state + /gantry/state -> /payload/relative_state"
        )

    def _gantry_state_cb(self, msg: GantryState):
        self.latest_gantry_state = msg

    def _payload_state_cb(self, msg: PayloadState):
        if self.latest_gantry_state is None:
            self.get_logger().warn_throttle(
                5.0,
                "Waiting for /gantry/state before publishing /payload/relative_state",
            )
            return

        if not msg.valid:
            return

        if not math.isfinite(msg.gantry_x) or not math.isfinite(msg.gantry_y):
            return

        out = PayloadState()

        # Copy header/timing.
        out.header = msg.header
        out.header.frame_id = "gantry_relative"
        out.motion_time_sec = msg.motion_time_sec

        # Keep legacy tag/camera fields unchanged.
        out.x1 = msg.x1
        out.z1 = msg.z1
        out.x2 = msg.x2
        out.z2 = msg.z2

        out.vx1 = msg.vx1
        out.vz1 = msg.vz1
        out.vx2 = msg.vx2
        out.vz2 = msg.vz2

        out.cam_x = msg.cam_x
        out.cam_y = msg.cam_y
        out.cam_z = msg.cam_z

        # Relative payload displacement wrt moving gantry/trolley.
        out.gantry_x = msg.gantry_x - self.latest_gantry_state.x
        out.gantry_y = msg.gantry_y - self.latest_gantry_state.y
        out.gantry_z = msg.gantry_z

        # Relative velocity wrt moving gantry/trolley.
        # GantryState vx/vy are m/s.
        out.v_gantry_x = msg.v_gantry_x - self.latest_gantry_state.vx
        out.v_gantry_y = msg.v_gantry_y - self.latest_gantry_state.vy
        out.v_gantry_z = msg.v_gantry_z

        out.valid = True
        out.interpolated = msg.interpolated
        out.frame_id = "gantry_relative"

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PayloadRelativeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
