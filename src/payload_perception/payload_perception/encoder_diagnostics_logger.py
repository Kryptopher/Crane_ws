#!/usr/bin/env python3
"""
encoder_diagnostics_logger — Record /payload/encoder/diagnostics to CSV.

Replaces manually running:
  ros2 topic echo /payload/encoder/diagnostics --field data
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray

_DEFAULT_FIELDS = (
    'time', 'arduino_ms', 'pitch_raw', 'roll_raw', 'pitch_count', 'roll_count',
    'imu1_gx_dps', 'imu1_gy_dps', 'imu1_gz_dps',
    'imu2_gx_dps', 'imu2_gy_dps', 'imu2_gz_dps',
    'packet_age_ms', 'packet_seen', 'serial_lines', 'parse_errors', 'stale',
    'wrap_events', 'sample_age_ms',
)

_OUTPUT_FIELDS = (
    'time', 'arduino_ms', 'pitch_count', 'roll_count',
    'imu1_gx_dps', 'imu1_gy_dps', 'imu1_gz_dps',
    'imu2_gx_dps', 'imu2_gy_dps', 'imu2_gz_dps',
    'packet_age_ms', 'packet_seen', 'serial_lines', 'parse_errors', 'stale',
    'wrap_events', 'sample_age_ms',
)


class EncoderDiagnosticsLogger(Node):
    def __init__(self):
        super().__init__('encoder_diagnostics_logger')

        self.declare_parameter('output_dir', os.path.expanduser('~/payload_logs'))
        self.declare_parameter('flush_every', 50)

        self._flush_every = int(self.get_parameter('flush_every').value)

        out_dir = Path(self.get_parameter('output_dir').value).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self._path = out_dir / f'encoder_diagnostics_{ts}.csv'

        self._fp = open(self._path, 'w', newline='')
        self._writer = csv.writer(self._fp)
        self._header_written = False
        self._rows = 0
        self._col_indices: Optional[List[int]] = None

        self.create_subscription(
            Float64MultiArray, '/payload/encoder/diagnostics',
            self._on_diag, qos_profile_sensor_data)

        self.get_logger().info(
            f'Logging /payload/encoder/diagnostics → {self._path}')

    def _on_diag(self, msg: Float64MultiArray):
        if not self._header_written:
            fields = list(_DEFAULT_FIELDS)
            if msg.layout.dim:
                label = msg.layout.dim[0].label
                if label:
                    fields = label.split(',')
            # Tolerate a publisher that predates a field (e.g. wrap_events).
            present = [name for name in _OUTPUT_FIELDS if name in fields]
            self._col_indices = [fields.index(name) for name in present]
            self._writer.writerow(present)
            self._header_written = True

        row: List[str] = [f'{msg.data[i]:.6f}' for i in self._col_indices]
        self._writer.writerow(row)
        self._rows += 1
        if self._rows % self._flush_every == 0:
            self._fp.flush()

    def destroy_node(self):
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node: Optional[EncoderDiagnosticsLogger] = None
    try:
        node = EncoderDiagnosticsLogger()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[encoder_diagnostics_logger] Fatal: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
