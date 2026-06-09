#!/usr/bin/env python3
"""
JOG position verification — compare encoder feedback to a manual tape measure.

Prerequisites:
  - Gantry running: ros2 launch gantry_control gantry.launch.py start_tracker:=false
  - Motors enabled + homed + JOG mode (see below)

Usage:
  ros2 run gantry_control jog_position_verify.py

Workflow:
  1. Note live (x, y) on screen — press Enter to mark START
  2. Jog with joystick (LB + left stick) or push cart slowly by hand if disabled
  3. Measure travel on the floor/rails (mm). Press Enter to mark END
  4. Script prints system delta (mm) vs your measured delta

Commands (gantry_cli in another terminal):
  ros2 run gantry_control gantry_cli.py enable
  ros2 run gantry_control gantry_cli.py home      # sets encoder origin at current pose
  ros2 run gantry_control gantry_cli.py jog 0     # slowest preset (~50 mm/s)
"""
import sys
import select
import time

import rclpy
from rclpy.node import Node
from gantry_control.msg import GantryState


class JogPositionVerify(Node):
    def __init__(self):
        super().__init__('jog_position_verify')
        self.latest: GantryState | None = None
        self.create_subscription(GantryState, '/gantry/state', self._cb, 10)
        self.start_xy = None
        self.msg_count = 0
        self.t0 = time.monotonic()

    def _cb(self, msg: GantryState):
        self.latest = msg
        self.msg_count += 1

    def wait_for_state(self, timeout_sec: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest is not None:
                return True
        return False

    def fmt_state(self, msg: GantryState) -> str:
        return (
            f"mode={msg.mode} enabled={msg.enabled} homed={msg.homed} | "
            f"x={msg.x * 1000:.1f} mm  y={msg.y * 1000:.1f} mm | "
            f"vx={msg.vx * 1000:.1f} vy={msg.vy * 1000:.1f} mm/s"
        )


def main():
    rclpy.init()
    node = JogPositionVerify()

    print("Waiting for /gantry/state ...")
    if not node.wait_for_state():
        print("No /gantry/state — is gantry_controller running? (USB hub: ls /dev/ttyXRUSB*)")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    dt = time.monotonic() - node.t0
    hz = node.msg_count / dt if dt > 0 else 0.0
    print(f"Connected (~{hz:.0f} Hz). Press Enter to mark START, move cart, Enter for END.")
    print("Ctrl+C to quit.\n")

    start_xy = None
    last_print = 0.0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if node.latest and now - last_print >= 0.5:
                last_print = now
                sys.stdout.write('\r' + node.fmt_state(node.latest) + '    ')
                sys.stdout.flush()

            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                if node.latest is None:
                    continue
                x, y = node.latest.x, node.latest.y
                if start_xy is None:
                    start_xy = (x, y)
                    print(
                        f"\n>>> START marked: ({x * 1000:.1f}, {y * 1000:.1f}) mm — jog now, then Enter"
                    )
                else:
                    dx = (x - start_xy[0]) * 1000
                    dy = (y - start_xy[1]) * 1000
                    dist = (dx * dx + dy * dy) ** 0.5
                    print(f"\n>>> END:   ({x * 1000:.1f}, {y * 1000:.1f}) mm")
                    print(f">>> System delta:  dx={dx:+.1f} mm  dy={dy:+.1f} mm  |Δ|={dist:.1f} mm")
                    print(">>> Compare to your tape measure along the same axis.")
                    print("    Press Enter again for another trial (new START).\n")
                    start_xy = None
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
