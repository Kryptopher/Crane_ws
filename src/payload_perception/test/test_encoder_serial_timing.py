import math
import threading
from types import SimpleNamespace

import pytest

import payload_perception.encoder_serial_node as encoder_module
from payload_perception.encoder_serial_node import EncoderSerialNode


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _velocity_only_node(alpha=0.35):
    node = object.__new__(EncoderSerialNode)
    node._prev_rel_t = None
    node._prev_rel_pos = None
    node._prev_rel_vel = None
    node._last_raw_rel_vel = None
    node._rel_vel_min_dt_s = 0.003
    node._rel_vel_alpha = alpha
    return node


def test_relative_velocity_exposes_raw_derivative_before_ema():
    node = _velocity_only_node()

    filtered, raw = node._payload_relative_velocity(1.0, (0.0, 0.0, 0.0))
    assert all(math.isnan(value) for value in filtered)
    assert all(math.isnan(value) for value in raw)

    filtered, raw = node._payload_relative_velocity(1.01, (0.01, 0.0, 0.0))
    assert raw[0] == pytest.approx(1.0)
    assert filtered[0] == pytest.approx(1.0)

    filtered, raw = node._payload_relative_velocity(1.02, (0.03, 0.0, 0.0))
    assert raw[0] == pytest.approx(2.0)
    assert filtered[0] == pytest.approx(0.35 * 2.0 + 0.65 * 1.0)


def test_publish_uses_serial_receive_time_and_does_not_repeat_pose(monkeypatch):
    node = object.__new__(EncoderSerialNode)
    node._lock = threading.Lock()
    node._arduino_ms = 1234.0
    node._pitch_raw = 10
    node._roll_raw = 20
    node._gyro_dps = [0.0] * 6
    node._packet_age_ms = 0.0
    node._packet_seen = 1.0
    node._serial_lines = 7
    node._parse_errors = 0
    node._wrap_events = 0
    node._last_rx_mono = 12.345
    node._stale_timeout_s = 0.5
    node._last_published_serial_lines = -1
    node._t0 = 10.0
    node._deg_per_count = 0.1
    node._relative_counts_locked = lambda: (10, 20)
    node.pub = _Publisher()
    node.pub_rel = _Publisher()
    node.pub_imu = _Publisher()
    node.pub_diag = _Publisher()
    node._build_rel_msg = lambda t, pitch, roll, age: SimpleNamespace(
        t=t, pitch=pitch, roll=roll, age=age)
    node._build_imu_msg = lambda *values: SimpleNamespace(values=values)
    monkeypatch.setattr(encoder_module.time, 'monotonic', lambda: 12.350)

    node._publish()

    assert len(node.pub.messages) == 1
    assert node.pub.messages[0].data[0] == pytest.approx(2.345)
    assert node.pub_rel.messages[0].age == pytest.approx(5.0)
    assert node.pub_diag.messages[0].data[-1] == pytest.approx(5.0)

    node._publish()

    assert len(node.pub.messages) == 1
    assert len(node.pub_rel.messages) == 1
    assert len(node.pub_imu.messages) == 1
    assert len(node.pub_diag.messages) == 2
