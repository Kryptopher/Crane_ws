#!/usr/bin/env python3
"""
Synthetic /payload/imu_raw publisher for logger/stack tests (no hardware).

  ros2 run payload_perception test_publish_imu_raw --duration 5 --rate 100

Publishes two sinusoidal IMU channels so the logger can exercise the IMU path.
"""
from __future__ import annotations

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

IMU_RAW_FIELDS = (
    'time', 'arduino_ms',
    'imu1_gx_dps', 'imu1_gy_dps', 'imu1_gz_dps',
    'imu2_gx_dps', 'imu2_gy_dps', 'imu2_gz_dps',
    'packet_age_ms', 'packet_seen',
)


class FakeImuRaw(Node):
    def __init__(self, rate_hz: float, duration: float):
        super().__init__('test_publish_imu_raw')
        self._pub = self.create_publisher(
            Float64MultiArray, '/payload/imu_raw', qos_profile_sensor_data)
        self._t0 = time.monotonic()
        self._duration = duration
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f'/payload/imu_raw publishing at {rate_hz:.0f} Hz for {duration:.1f}s')

    def _tick(self):
        t = time.monotonic() - self._t0
        if t > self._duration:
            raise SystemExit(0)

        gx1 = 10.0 * math.sin(2 * math.pi * 0.5 * t)
        gy1 = 10.0 * math.cos(2 * math.pi * 0.5 * t)
        gz1 = 5.0 * math.sin(2 * math.pi * 0.3 * t)
        gx2, gy2, gz2 = gx1, gy1, gz1

        msg = Float64MultiArray()
        dim = MultiArrayDimension()
        dim.label = ','.join(IMU_RAW_FIELDS)
        dim.size = len(IMU_RAW_FIELDS)
        dim.stride = len(IMU_RAW_FIELDS)
        msg.layout.dim.append(dim)
        msg.data = [
            float(t), float(t * 1000),
            gx1, gy1, gz1, gx2, gy2, gz2,
            0.0, 1.0,
        ]
        self._pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rate', type=float, default=100.0,
                        help='Publish rate in Hz (default 100)')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='How long to publish in seconds (default 5)')
    args = parser.parse_args()

    rclpy.init()
    node = None
    try:
        node = FakeImuRaw(args.rate, args.duration)
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
