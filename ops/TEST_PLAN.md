# Systematic test plan — crane_ws

Run automated phases first (no hardware), then manual phases with OAK + crane.

## Setup (every terminal)

```bash
source /opt/ros/humble/setup.bash
source ~/crane_ws/install/setup.bash
cd ~/crane_ws/ops
```

## Automated runner

```bash
chmod +x test_stack.sh
./test_stack.sh all      # phases 0–2 automated + 3–4 checklists
./test_stack.sh 1        # traj logging only
./test_stack.sh 2        # PayloadState + logger only
```

Exit code 0 = all automated checks passed.

---

## Phase 0 — Build & environment

| Check | Command | Pass |
|-------|---------|------|
| Packages built | `ros2 pkg list \| grep -E 'gantry_control\|payload_perception\|experiment_logger'` | 3 packages |
| PayloadState msg | `ros2 interface show payload_perception/msg/PayloadState` | shows vx1..vz2 |
| Executables | `ros2 pkg executables payload_perception` | includes `test_publish_payload_state` |

---

## Phase 1 — Traj logging (no USB, no camera)

Validates **control + logging** path only.

| Step | Command |
|------|---------|
| T1 | `./run_traj_log.sh logonly` |
| T2 | `./run_traj_log.sh play --no-arm ~/csv_profiles/pulse.csv` |
| T2 | `./run_traj_log.sh show` |

| Pass criteria |
|---------------|
| `Lines: 7` or more on latest `logger_traj_cmd_*.csv` |
| Rows include `PROFILE_START`, `WAYPOINT`, `PROFILE_DONE` |

---

## Phase 2 — PayloadState + logger (no camera)

Validates **pose + velocity** topic and CSV columns.

| Step | Command |
|------|---------|
| T1 | `ros2 run experiment_logger logger_node --ros-args -p wait_motion_start:=false` |
| T2 | `ros2 run payload_perception test_publish_payload_state --duration 3` |

Or: `./test_stack.sh 2`

| Pass criteria |
|---------------|
| New `logger_sync_pose_*.csv` |
| Header contains `vx1,vz1,vx2,vz2` |
| ≥ ~100 data rows (3 s @ 50 Hz) |
| `vx1` column not all zero |

**Optional topic inspection:**

```bash
ros2 topic echo /payload/state --once
ros2 topic echo /payload/pose --once   # legacy 9-element array
```

---

## Phase 3 — Live vision (OAK required)

| Step | Command |
|------|---------|
| T1 | `ros2 run payload_perception payload_tracker --ros --ip 192.168.0.153` |
| T2 | `ros2 topic hz /payload/state` |
| T2 | `ros2 topic echo /payload/state --once` |

| Pass criteria |
|---------------|
| `hz` roughly matches detect FPS (often 30–120) |
| `valid: true` when tags in view |
| Positions finite (not NaN) |
| Velocities non-zero when you move the payload by hand |
| Web stream: `http://192.168.0.101:8080/` |

**MOTION_START gate test:**

1. Launch with `gantry.launch.py` (tracker has `--wait-motion`).
2. Before `enable`: `ros2 topic hz /payload/state` → **0 Hz**.
3. After `gantry_cli enable`: hz > 0, `motion_time_sec` starts near 0.

---

## Phase 4 — Full experiment (USB + OAK)

| Step | Command |
|------|---------|
| T1 | `ros2 launch gantry_control gantry.launch.py oak_ip:=192.168.0.153` |
| T2 | `ros2 run gantry_control traj_player.py ~/csv_profiles/pulse.csv` |
| T2 | `ros2 run gantry_control gantry_cli.py enable` |

| Pass criteria |
|---------------|
| `logger_traj_cmd_*.csv`: PLAYBACK rows @ ~100 Hz |
| `logger_sync_pose_*.csv`: pose + velocity after MOTION_START |
| `ros2 topic echo /gantry/state` → `|x|`, `|y|` ≤ 1.0 m |
| No unexpected E-STOP |

**Realtime TRAJ (optional):**

```bash
# set_mode TRAJ, enable (no profile), then:
ros2 run gantry_control traj_stream_example.py --vy 100 --duration 3
```

---

## Regression matrix

| Feature | Phase |
|---------|-------|
| Buffered traj_player | 1 |
| PayloadState + velocities | 2 |
| MOTION_START sync | 3 |
| Workspace limits | 4 |
| Legacy `/payload/pose` 9-array | 2 (echo once) |

---

## Failure triage

| Symptom | Likely cause |
|---------|----------------|
| Empty traj CSV | Logger not running; traj_player before subscriber match |
| No vx columns | Old logger build; rebuild `experiment_logger` |
| No /payload/state | Tracker not `--ros`; still waiting for MOTION_START |
| All vx = 0 | Tags static; move payload or check dz_dt / fallback |
| Package not found | Forgot `source ~/crane_ws/install/setup.bash` |
