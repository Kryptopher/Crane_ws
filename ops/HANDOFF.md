# crane_ws — project handoff

**Paste this file (or `ops/PROJECT_HANDOFF.txt`) when starting a new chat.**

| Item | Value |
|------|--------|
| Jetson user | `sanjay` |
| Jetson IP | `192.168.0.101` — SSH, MJPEG `:8080`, dashboard `:8081` |
| OAK camera IP | `192.168.0.153` — **not** the Jetson |
| Workspace | `/home/sanjay/crane_ws` |
| Logs | `~/payload_logs` |
| Trajectories | `~/csv_profiles/` (`time_s, vx_mm_s, vy_mm_s`) |
| Legacy (ignore) | `~/clearpath_ws`, `~/payload_tracker_archive_*` |

---

## Shell setup (every terminal)

```bash
source /opt/ros/humble/setup.bash
source ~/crane_ws/install/setup.bash
```

Rebuild after code/msg changes:

```bash
cd ~/crane_ws && colcon build --symlink-install
source install/setup.bash
```

Typical selective rebuild:

```bash
colcon build --packages-select payload_perception_msgs payload_perception gantry_control --symlink-install
```

---

## Packages

| Package | Role |
|---------|------|
| `gantry_control` | ClearPath gantry, TRAJ, web UI, launches, `adaptive_id_player` |
| `payload_perception` | OAK AprilTag `payload_tracker`, optional `encoder_node`, `payload_gantry_frame` (cart-mounted cam only) |
| `payload_perception_msgs` | `PayloadState.msg` |
| `experiment_logger` | CSV logger for `/traj_cmd` + pose streams |

---

## Lab geometry (critical)

- **OAK is on a fixed tripod** in front of the payload — **not** on the moving cart.
- **Do not use** `payload_gantry_frame` or `start_gantry_frame:=true` (that node subtracts cart motion for a camera on the cart).
- Vision → gantry is **fixed mount rotation only** in `payload_tracker` (`--gantry-yaw-deg`, pitch, roll, signs).
- See **`ops/PAYLOAD_FRAMES.md`** for axis mapping (cam X/Z horizontal, cam Y vertical; gantry X/Y cart, gantry Z swing).

| Camera view (tripod) | Tracker fields |
|--------------------|----------------|
| Swing left/right | `cam_x` → calibrate to `gantry_z` |
| Cart X+ → tag moves left | `cam_x` → `gantry_x` (often `--gantry-sign-x -1`) |
| Cart Y+ → closer | `cam_z` → `gantry_y` |
| Up/down in image | `cam_y` (usually small) |

---

## Three ways to run

### 1) Payload tracker only (vision debug)

```bash
cd ~/crane_ws && source install/setup.bash
./ops/run_payload_tracker.sh
```

- MJPEG: `http://192.168.0.101:8080/` (camera + **TOP-DOWN** panel on the right)
- ROS: `/payload/state`, `/payload/pose`, `/payload/camera/compressed`
- Uses **`--no-wait-motion`** so topics publish immediately (no gantry enable needed)

Verify:

```bash
ros2 topic hz /payload/state
ros2 topic echo /payload/state --once
```

### 2) Full gantry stack (no dashboard)

```bash
ros2 launch gantry_control gantry.launch.py oak_ip:=192.168.0.153
```

- Tracker: `--no-wait-motion`, smoothing args in launch
- Vision-only: `start_gantry:=false start_joy:=false`

### 3) Mission planner + crane ops UI

```bash
ros2 launch gantry_control mission_planner.launch.py oak_ip:=192.168.0.153
```

- Dashboard: **`http://192.168.0.101:8081/`** (`crane_ops.html`)
- Raw stream: `:8080`
- Details: **`ops/MISSION_PLANNER.md`**

---

## Key ROS topics

| Topic | Type | Notes |
|-------|------|--------|
| `/payload/state` | `payload_perception_msgs/PayloadState` | **Primary** — `cam_*`, `gantry_*`, per-tag `x1/z1`, `motion_time_sec` |
| `/payload/pose` | `Float64MultiArray` | Legacy `[time, x1, z1, x2, z2, vx1, vz1, vx2, vz2]` |
| `/payload/pose_e` | `Float64MultiArray` | GPIO encoders `[time, pitch_deg, roll_deg, ...]` — rate follows `/stack/pose_sync_hz` |
| `/stack/pose_sync_hz` | `std_msgs/Float64` | Vision-led stack sync rate (EMA of detect FPS) |
| `/payload/camera/compressed` | `sensor_msgs/CompressedImage` | Annotated HD frame (dashboard) |
| `/gantry/state` | `GantryState` | Cart x, y, mode, enabled, estop |
| `/traj_cmd` | `TrajCmd` | Profile, MOTION_START, PLAYBACK, STREAM |

### Publish rates (vision-led adaptive sync)

| Topic | Rate | Mechanism |
|-------|------|-----------|
| `/stack/pose_sync_hz` | EMA of measured **detect FPS** | `payload_tracker` publishes every 0.5 s |
| `/payload/state` | **= detect FPS** | Per-frame publish when `--sync-adaptive` (mission default) |
| `/gantry/state` | follows sync topic | `stack_pose_sync_adaptive: true` in `gantry.yaml` |
| `/payload/pose_e` | follows sync topic | `follow_sync_hz: true` in `config/encoder.yaml` |
| Control loop | **100 Hz** | `gantry_controller` jog/homing/mission (unchanged) |

Mission throughput: HD OAK branch off (`--no-stream`), detect-res dashboard JPEG @ 15 Hz, `--turbo`, `quad_decimate` 2.5.

Verify on Jetson:

```bash
./ops/verify_pose_rates.sh 10
```

Dashboard `/state` reports `stack_sync_hz`, `payload_hz`, `gantry_hz`, `pose_e_hz`, and `pose_rates_synced` (within ~3 Hz of `stack_sync_hz`).

`PayloadState` (session-relative m, after first detection):

- `cam_x`, `cam_y`, `cam_z` — camera/tripod frame  
- `gantry_x`, `gantry_y`, `gantry_z` — crane frame (`frame_id: gantry`)  
- `x1`, `z1`, `x2`, `z2` — per-tag camera samples (legacy names)

---

## Payload tracker — smoothing & TOP-DOWN plot

**ROS smoothing** (reduces jitter on `/payload/state`):

1. Median + EMA on raw tag center before session zero  
2. One publish per main loop (no burst spam)  
3. Median + EMA on `cam_*` (`--publish-alpha`, default `0.14`)  
4. Median + EMA on `gantry_*` after rotation (`--gantry-publish-alpha`, default `0.16`)  
5. Spike clamp `--publish-max-step-m` (default `0.010` m/frame)

**TOP-DOWN panel** (in MJPEG `:8080`, class `TopDownView`):

- Auto-scales X/Z; always shows origin **0** with crosshair  
- Default Z window `--z-min -0.6` `--z-max 1.2` (was `0..3`, which clipped negative Z)  
- Extra bottom margin so marker is not cut off  

Tuning examples:

```bash
# Smoother / slower
--publish-alpha 0.10 --gantry-publish-alpha 0.12 --publish-max-step-m 0.006

# More TOP-DOWN vertical range
--z-min -0.9 --z-max 1.0
```

Debug AprilTag spam: `--debug-detect` (off by default).

---

## TRAJ workflow (buffered profile)

1. Load profile (motors off):

   ```bash
   ros2 run gantry_control traj_player.py ~/csv_profiles/pulse.csv
   ```

2. Enable (t=0, MOTION_START, motion + logging):

   ```bash
   ros2 run gantry_control gantry_cli.py enable
   ```

3. Optional log-only: `traj_player.py --no-arm ...`

**TrajCmd commands:** `WAYPOINT=0`, `PROFILE_DONE=1`, `ABORT=2`, `PROFILE_START=3`, `MOTION_START=4`, `PLAYBACK=5`, `STREAM=6`.

**Realtime TRAJ:** `set_mode TRAJ` → enable → `traj_stream_example.py` @ ~100 Hz.

---

## Adaptive system ID

```bash
ros2 run gantry_control adaptive_id_player.py
```

Defaults: `/payload/state`, `payloadstate`, swing field **`gantry_z`**.  
Until mount calibration is good: `--swing-field cam_x`.

Syncs integrators on `/traj_cmd` `MOTION_START` after enable.

---

## Safety

| Feature | Status |
|---------|--------|
| Workspace limit 1.0 m (motor encoders) | **On** (`gantry.yaml`) |
| Payload vision E-stop | **Off** (`payload_limit_monitor_enable: false`) until `gantry_*` trusted |
| Ctrl+C shutdown | Disables motors, no E-stop latch (`shutdown_for_exit`) |

Clear E-stop: `ros2 run gantry_control gantry_cli.py clear_estop`

---

## Gantry sensor homing (SC4-HUB inputs — no ClearView homing save)

Conductive home sensors on the SC4-HUB (USB to Jetson). Default mode **`homing_mode: sensor_inputs`** in `gantry.yaml` — ROS reads digital inputs and seeks **back then left** until both are active. **No Teknic homing profile** needs to be saved in ClearView (sensors can be moved; tune YAML only).

| Sensor | Hub connector | Software node | Input |
|--------|---------------|---------------|-------|
| Back | **0A** | 0 (motor A / CP0) | A | Hub LED **off** at home (`home_back_input_active_high: false`) |
| Left | **1B** | 1 (motor B / CP1) | B | Hub LED **on** at home (`home_left_input_active_high: true`) |

Sensors: OMCH SN04-N2 (NPN inductive, ~5 mm). NPN often pulls low when metal is present; Teknic may show that as input inactive on 0A (green off at home).

Communication path: Jetson → USB → SC4-HUB → each ClearPath node; `gantry_controller` polls `node->Status.RT` (`InA` / `InB`).

**Sequence:** slow **−Y** until back (**0A**) → slow **−X** until left (**1B**) → snapshot encoders → `homed=true`. If already at home, finalize immediately.

**Parameters (`gantry_control/config/gantry.yaml`):**

- `homing_mode: sensor_inputs` (or `teknic_homing` for legacy ClearView `Homing.Initiate`)
- `homing_seek_vel_ms`, `home_input_active_high`, `homing_input_debounce_ticks`
- `auto_home_before_run: true` — before TRAJ / MISSION / CSV runs

**Automated test (USB on Jetson, not laptop):**

```bash
source /opt/ros/humble/setup.bash && source ~/crane_ws/install/setup.bash
ros2 launch gantry_control gantry.launch.py start_encoder:=false start_tracker:=false
# other terminal:
ros2 run gantry_control gantry_cli.py enable
ros2 run gantry_control gantry_cli.py home
ros2 topic echo /gantry/state   # homing_status: seek_back, seek_left, done
```

ClearView on a laptop is optional (watch hub input LEDs while jogging). For `teknic_homing`, ClearView `HOME_TO_SWITCH` setup is required per motor.

---

## Gantry odometry

After homing, cart pose is in the **gantry** frame (lab: origin bottom-left, +X right, +Y up):

| Output | Topic / TF |
|--------|------------|
| `nav_msgs/Odometry` | `/gantry/odom` — `header.frame_id=gantry`, `child_frame_id=gantry_cart` |
| TF | `gantry` → `gantry_cart` (same pose; enable with `publish_odom_tf`) |

Position/velocity match `/gantry/state` `x`, `y`, `vx`, `vy` (meters, homed encoder offsets). Planar: `z=0`, identity orientation.

```bash
ros2 topic echo /gantry/odom
ros2 run tf2_ros tf2_echo gantry gantry_cart
```

---

## Two encoder systems (do not confuse)

1. **Motor encoders** — `gantry_controller` → cart X/Y, TRAJ PLAYBACK  
2. **Payload GPIO encoders** — `encoder_node` → `/payload/pose_e` → logger / dashboard pendulum UI only  

Payload harness (Jetson **BOARD** pins; 40-pin header GPIO names):

| Axis | ROS param | A | B | Header |
|------|-----------|---|---|--------|
| Pitch | `enc1_a` / `enc1_b` | 33 | 32 | GPIO13 / GPIO07 |
| Roll | `enc2_a` / `enc2_b` | 29 | 31 | GPIO01 / GPIO11 |

Params: `gantry_control/config/experiment.yaml`. Bench: `python3 ~/crane_ws/ops/encoder_reader.py`.
Launch uses `ros2 run payload_perception encoder_node` (see `gantry.launch.py`).

---

## Important files

```
crane_ws/
  ops/HANDOFF.md              ← this document
  ops/PAYLOAD_FRAMES.md       ← camera/gantry axes
  ops/MISSION_PLANNER.md      ← dashboard launch
  ops/TRAJ_RUNBOOK.md
  ops/run_payload_tracker.sh  ← tracker-only test
  ops/test_stack.sh
  src/gantry_control/
    gantry_control/web/crane_ops.html
    gantry_control/scripts/crane_dashboard_server.py
    gantry_control/scripts/adaptive_id_player.py
    gantry_control/launch/mission_planner.launch.py
    gantry_control/launch/gantry.launch.py
    config/gantry.yaml
  src/payload_perception/
    payload_perception/payload_tracker.py
    payload_perception/payload_frames.py
    payload_perception/payload_gantry_frame.py  ← do not use (tripod cam)
```

---

## Logs (~/payload_logs)

| File | Content |
|------|---------|
| `logger_traj_cmd_*.csv` | Traj + motor encoders @ 100 Hz |
| `logger_sync_pose_*.csv` | Vision x1/z1 and/or encoder angles after MOTION_START |
| `vision_osc_*.csv` | Tracker internal filtered X + FFT |
| `vision_tags_*.csv` | Per-tag positions |

Download (Windows PowerShell):

```powershell
scp sanjay@192.168.0.101:~/payload_logs/* $env:USERPROFILE\Downloads\jetson_logs\
```

---

## Common failures

| Symptom | Fix |
|---------|-----|
| `Package not found` | `source ~/crane_ws/install/setup.bash` |
| OAK `DEVICE_NOT_FOUND` | `oak_ip:=192.168.0.153` (camera IP, not Jetson) |
| No `/payload/state` | Tracker needs `--no-wait-motion` for standalone test |
| No `/payload/pose_e` | `start_encoder:=true` on launch; `pkill -f encoder_reader`; check `ros2 topic hz /payload/pose_e` |
| Port 8080 busy | `pkill -f payload_tracker` |
| No ClearPath USB | `start_gantry:=false` or plug `/dev/ttyXRUSB*` |
| GPIO busy (encoder) | `pkill -f encoder_node; pkill -f encoder_reader` |
| Dashboard PREVIEW/mock | `/state` not reachable — check `crane_dashboard_server` |
| TOP-DOWN marker clipped | Fixed in current `TopDownView`; restart tracker |

---

## Done recently (May 2026)

- `crane_ws` unified stack; tripod camera documented; cart compensation disabled  
- `PayloadState` with `cam_*` / `gantry_*`; tracker does not subscribe to `/gantry/state`  
- Pose smoothing + coalesced ROS publish; TOP-DOWN auto-bounds + visible zero  
- Mission planner UI: Enable/Home/E-stop, JOG mode sync fixes, vision plot smoothing  
- `adaptive_id_player` → `/payload/state` + `gantry_z`  
- `ops/run_payload_tracker.sh` for isolated vision tests  

---

## Next / deferred

- [ ] Calibrate `--gantry-yaw-deg` / pitch / roll / signs (swing → `gantry_z`) — save in `config/payload_mount.yaml`  
- [ ] Re-enable payload vision limits in gantry frame when pose is trusted  
- [x] Logger CSV: `cam_*`, `gantry_*`, `cart_x/y` from `/gantry/state`  
- [ ] Axis encoder calibration (user deferred)  
- [x] `[DETECT-DBG]` gated behind `--debug-detect`  
- [x] `/payload/state`, `/gantry/state`, `/payload/pose_e` follow `/stack/pose_sync_hz` — verify with `./ops/verify_pose_rates.sh`  

---

## Quick reference — paste for new agent

```
Workspace: ~/crane_ws
Jetson: 192.168.0.101  OAK: 192.168.0.153
OAK on TRIPOD (fixed) — no payload_gantry_frame
Pose: /payload/state (PayloadState), gantry_* from tracker rotation only
Tracker only: ./ops/run_payload_tracker.sh  → :8080 + /payload/state
Full UI: ros2 launch gantry_control mission_planner.launch.py → :8081
Frames doc: ops/PAYLOAD_FRAMES.md
```
