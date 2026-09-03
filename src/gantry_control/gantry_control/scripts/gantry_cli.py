#!/usr/bin/env python3
"""
Gantry CLI — quick command-line interface for common operations.

Usage:
    python3 gantry_cli.py enable
    python3 gantry_cli.py home
    python3 gantry_cli.py force-home         # set current position as home (no seek)
    python3 gantry_cli.py jog [0|1|2]       # speed preset
    python3 gantry_cli.py csv <path>
    python3 gantry_cli.py traj          # arm TRAJ (run traj_player.py with velocity CSV)
    python3 gantry_cli.py mission
    python3 gantry_cli.py goto <x_m> <y_m>
    python3 gantry_cli.py stop
    python3 gantry_cli.py estop
    python3 gantry_cli.py clear
    python3 gantry_cli.py disable
    python3 gantry_cli.py status
"""
import sys
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from gantry_control.srv import SetMode, MoveTo
from gantry_control.msg import GantryState


class GantryCLI(Node):
    def __init__(self):
        super().__init__('gantry_cli')

    def call_trigger(self, service_name):
        cli = self.create_client(Trigger, service_name)
        if not cli.wait_for_service(timeout_sec=8.0):
            print(f"Service {service_name} not available.")
            return
        req = Trigger.Request()
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result():
            r = future.result()
            print(f"{'OK' if r.success else 'FAIL'}: {r.message}")
        else:
            print("Service call failed.")

    def call_set_mode(self, mode, csv_path='', jog_preset=0, tx=0.0, ty=0.0):
        cli = self.create_client(SetMode, '/gantry/set_mode')
        if not cli.wait_for_service(timeout_sec=8.0):
            print("Service /gantry/set_mode not available.")
            return
        req = SetMode.Request()
        req.mode = mode
        req.csv_path = csv_path
        req.jog_speed_preset = jog_preset
        req.target_x = tx
        req.target_y = ty
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result():
            r = future.result()
            print(f"{'OK' if r.success else 'FAIL'}: {r.message}")
        else:
            print("Service call failed.")

    def call_move_to(self, x, y):
        cli = self.create_client(MoveTo, '/gantry/move_to')
        if not cli.wait_for_service(timeout_sec=8.0):
            print("Service /gantry/move_to not available.")
            return
        req = MoveTo.Request()
        req.x = x
        req.y = y
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result():
            r = future.result()
            print(f"{'OK' if r.success else 'FAIL'}: {r.message}")
        else:
            print("Service call failed.")

    def print_status(self):
        """Subscribe to /gantry/state and print one message."""
        received = [False]
        def cb(msg):
            print(f"═══════════════════════════════════")
            print(f"  Mode:     {msg.mode}")
            print(f"  Enabled:  {msg.enabled}")
            print(f"  Homed:    {msg.homed}")
            print(f"  E-Stop:   {msg.estop}")
            print(f"  Position: ({msg.x*1000:.1f}, {msg.y*1000:.1f}) mm")
            print(f"  Velocity: ({msg.vx*1000:.1f}, {msg.vy*1000:.1f}) mm/s")
            print(f"  MoveDone: {msg.move_done}")
            print(f"  Timing:   read={msg.read_time_ms:.1f}ms write={msg.write_time_ms:.1f}ms")
            print(f"  Errors:   {msg.error_count}")
            print(f"═══════════════════════════════════")
            received[0] = True

        sub = self.create_subscription(GantryState, '/gantry/state', cb, 1)
        timeout = self.create_rate(10)
        for _ in range(30):  # wait up to 3 seconds
            rclpy.spin_once(self, timeout_sec=0.1)
            if received[0]:
                break
        if not received[0]:
            print("No state message received.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    rclpy.init()
    node = GantryCLI()

    cmd = sys.argv[1].lower()

    if cmd == 'enable':
        node.call_trigger('/gantry/enable')
    elif cmd == 'disable':
        node.call_trigger('/gantry/disable')
    elif cmd == 'estop':
        node.call_trigger('/gantry/estop')
    elif cmd == 'clear':
        node.call_trigger('/gantry/clear_estop')
    elif cmd == 'stop':
        node.call_set_mode('IDLE')
    elif cmd == 'home':
        node.call_set_mode('HOME')
    elif cmd in ('force-home', 'force_home', 'sethome', 'set-home'):
        node.call_trigger('/gantry/force_home')
    elif cmd == 'jog':
        preset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        node.call_set_mode('JOG', jog_preset=preset)
    elif cmd == 'csv':
        if len(sys.argv) < 3:
            print("Usage: gantry_cli.py csv <path_to_csv>")
        else:
            node.call_set_mode('CSV', csv_path=sys.argv[2])
    elif cmd == 'traj':
        node.call_set_mode('TRAJ')
        print('TRAJ armed. Run: ros2 run gantry_control traj_player.py <velocity.csv>')
        print('  Then: gantry_cli.py enable  (starts motion if profile loaded)')
    elif cmd == 'mission':
        node.call_set_mode('MISSION')
    elif cmd == 'goto':
        if len(sys.argv) < 4:
            print("Usage: gantry_cli.py goto <x_meters> <y_meters>")
        else:
            x = float(sys.argv[2])
            y = float(sys.argv[3])
            node.call_move_to(x, y)
    elif cmd == 'status':
        node.print_status()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
