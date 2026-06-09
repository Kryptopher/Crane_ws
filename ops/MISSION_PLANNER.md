# Mission planner (crane ops UI)

## Launch full stack + dashboard

```bash
cd ~/crane_ws
source install/setup.bash
export SDL_JOYSTICK_DEVICE=/dev/input/js0   # if not set by launch

ros2 launch gantry_control mission_planner.launch.py
```

Open in a browser (Jetson LAN IP):

- **Dashboard:** `http://192.168.0.101:8081/` — camera via `/payload/camera/compressed` (multipart `/camera/stream`)
- **Center plot:** gantry cart X/Y only (motors). **Vision chart:** payload swing (`gantry_x` vs `gantry_z`).
- **Reset vision origin** button on the dashboard (or `POST /action/reset_vision`) clears session zero + trail.
- **Raw MJPEG `:8080`:** only if tracker started with `--stream-port` (default launch uses `--no-stream`).

## What starts

| Node | Role |
|------|------|
| `gantry_controller` | USB gantry, `/gantry/state` |
| `joy_node` | Joystick JOG |
| `payload_tracker` | OAK → `/payload/state`, `/payload/camera/compressed` |
| `encoder_node` | `/payload/pose_e` — payload GPIO encoders (BOARD 33/32/29/31) |
| `crane_dashboard_server` | Serves `crane_ops.html`, ROS bridge `:8081` |

## Launch args

```bash
ros2 launch gantry_control mission_planner.launch.py \
  oak_ip:=192.168.0.153 \
  start_encoder:=true \
  start_tracker:=true \
  start_logger:=true \
  dashboard_port:=8081 \
  gantry_yaw_deg:=0.0 \
  gantry_sign_x:=-1.0
```

Vision-only bench (no motors):

```bash
ros2 launch gantry_control mission_planner.launch.py \
  start_gantry:=false start_joy:=false start_encoder:=false \
  oak_ip:=192.168.0.153
```

Disable encoder if harness not connected:

```bash
ros2 launch gantry_control mission_planner.launch.py start_encoder:=false
```

If encoder fails with **GPIO busy**, stop any other GPIO user (not `encoder_reader*.py` — use `encoder_node` only):

```bash
pkill -f encoder_node
pkill -f encoder_reader
pkill -f payload_tracker   # frees MJPEG port 8080
# then relaunch
```

Payload encoder wiring (Jetson **BOARD** pins; header GPIO names):

| Axis | A pin | B pin | Header |
|------|-------|-------|--------|
| Pitch | 33 | 32 | GPIO13 / GPIO07 |
| Roll | 29 | 31 | GPIO01 / GPIO11 |

Bench test (no ROS): `python3 ~/crane_ws/ops/encoder_reader.py` (stop `encoder_node` first).

`encoder_node` uses the same decode as the bench script plus ROS publishing, PPR degree scaling
(`counts_are_degrees: false` in `experiment.yaml`), optional `wait_motion_start` for logs,
and `interrupt_on_b: true` (IRQ on all four lines).

## Gantry sensor homing

Limit switches on the **ClearPath motors** (SC4-HUB USB). Configure homing in **Teknic ClearView** per motor (`HOME_TO_SWITCH`). With `auto_home_before_run: true` in `gantry.yaml`, the gantry homes automatically before **▶ Execute** (mission) and before **TRAJ** playback on enable — no separate Home click required (manual Home still available).

Watch `/gantry/state`: `homing_active`, `homing_status` (`motor_a`, `motor_b`, `done`, `failed`). See `ops/HANDOFF.md` for ClearView checklist.

## Typical workflow

1. **Enable** (sensor homing runs automatically before the first mission or TRAJ run).
2. Select **JOG**, pick speed preset; use L1 + stick (no auto-home per stick move).
3. For waypoints: **MISSION** or **CSV**, click canvas to add WPs, **▶ Execute** (homes first if configured).
4. Single point: **MISSION** panel → target X/Y → **Move to position**.

## Dashboard only (gantry already running)

```bash
ros2 run gantry_control crane_dashboard_server.py
```
