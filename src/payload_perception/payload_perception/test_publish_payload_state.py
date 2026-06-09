#!/usr/bin/env python3
"""
Synthetic /payload/state publisher for stack tests (no camera).

  ros2 run payload_perception test_publish_payload_state --duration 3
"""
from __future__ import annotations

import argparse
import math
import time

import rclpy
from payload_perception_msgs.msg import PayloadState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class FakePayloadState(Node):
    def __init__(self, rate_hz: float, duration: float):
        super().__init__('test_publish_payload_state')
        self._pub = self.create_publisher(
            PayloadState, '/payload/state', qos_profile_sensor_data)
        self._t0 = time.monotonic()
        self._duration = duration
        self._period = 1.0 / rate_hz
        self._timer = self.create_timer(self._period, self._tick)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._pub.get_subscription_count() > 0:
                self.get_logger().info(
                    f'/payload/state subscribers: {self._pub.get_subscription_count()}')
                break
        else:
            self.get_logger().warn('No subscribers on /payload/state')

    def _tick(self):
        t = time.monotonic() - self._t0
        if t > self._duration:
            self._publish(t, 0.0, 0.0, final=True)
            raise SystemExit(0)
        x1 = 0.05 * math.sin(2.0 * math.pi * 0.5 * t)
        z1 = 0.10 * math.cos(2.0 * math.pi * 0.5 * t)
        x2 = x1 + 0.02
        z2 = z1 + 0.03
        vx1 = 0.05 * math.pi * math.cos(2.0 * math.pi * 0.5 * t)
        vz1 = -0.10 * math.pi * math.sin(2.0 * math.pi * 0.5 * t)
        self._publish(t, x1, z1, x2, z2, vx1, vz1, vx1 * 0.9, vz1 * 0.9)

    def _publish(
        self, t, x1, z1, x2=0.0, z2=0.0,
        vx1=0.0, vz1=0.0, vx2=0.0, vz2=0.0, final=False,
    ):
        msg = PayloadState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'test'
        msg.motion_time_sec = t
        msg.x1, msg.z1, msg.x2, msg.z2 = x1, z1, x2, z2
        msg.vx1, msg.vz1, msg.vx2, msg.vz2 = vx1, vz1, vx2, vz2
        msg.valid = not final
        msg.interpolated = False
        msg.frame_id = 'test'
        self._pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rate', type=float, default=50.0)
    parser.add_argument('--duration', type=float, default=3.0)
    args = parser.parse_args()
    rclpy.init()
    node = None
    try:
        node = FakePayloadState(args.rate, args.duration)
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
