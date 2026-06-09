#!/usr/bin/env python3
"""
Example real-time TRAJ publisher — sends STREAM commands at 100 Hz.

Usage (gantry in TRAJ mode, motors enabled):
  ros2 run gantry_control traj_stream_example.py --vx 0 --vy 200

Stop: Ctrl+C or --duration 5
"""
from __future__ import annotations

import argparse
import time

import rclpy
from gantry_control.msg import TrajCmd
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class TrajStreamExample(Node):
    def __init__(self, vx_mm_s: float, vy_mm_s: float, rate_hz: float):
        super().__init__('traj_stream_example')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self._pub = self.create_publisher(TrajCmd, '/traj_cmd', qos)
        self._vx = vx_mm_s
        self._vy = vy_mm_s
        self._period = 1.0 / rate_hz
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._pub.get_subscription_count() > 0:
                break
        else:
            self.get_logger().warn('No /traj_cmd subscribers yet')
        self._timer = self.create_timer(self._period, self._tick)

    def _tick(self):
        msg = TrajCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = TrajCmd.STREAM
        msg.vx_mm_s = self._vx
        msg.vy_mm_s = self._vy
        self._pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vx', type=float, default=0.0, help='vx mm/s')
    parser.add_argument('--vy', type=float, default=0.0, help='vy mm/s')
    parser.add_argument('--rate', type=float, default=100.0, help='publish Hz')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='seconds (0 = run until Ctrl+C)')
    args = parser.parse_args()

    rclpy.init()
    node = TrajStreamExample(args.vx, args.vy, args.rate)
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = TrajCmd()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.command = TrajCmd.STREAM
        stop.vx_mm_s = 0.0
        stop.vy_mm_s = 0.0
        node._pub.publish(stop)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
