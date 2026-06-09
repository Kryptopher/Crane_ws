# Trajectory + logger — full runbook

## Fix your shell (one time)

Your login shows `not found: "/home/sanjay/install/local_setup.bash"`.  
Edit `~/.bashrc` and remove or fix any line that sources `~/install/local_setup.bash`.  
Use only:

```bash
source /opt/ros/humble/setup.bash
source ~/crane_ws/install/setup.bash
```

---

## Every new SSH terminal (T1, T2, T3)

Copy-paste this **before anything else**:

```bash
source /opt/ros/humble/setup.bash
source ~/crane_ws/install/setup.bash
cd ~/crane_ws/ops
```

If `ros2 run gantry_control ...` says **Package not found**, you skipped `source ~/crane_ws/install/setup.bash`.

---

## Path A — Test logging only (no crane USB)

Use this to verify CSV → `/traj_cmd` → logger → download. **No motors.**

**Terminal 1 (leave running):**

```bash
source /opt/ros/humble/setup.bash
source ~/crane_ws/install/setup.bash
cd ~/crane_ws/ops
./run_traj_log.sh logonly
```

Wait for: `Traj CSV ready ... logger_traj_cmd_*.csv`

**Terminal 2:**

```bash
source /opt/ros/humble/setup.bash
source ~/crane_ws/install/setup.bash
cd ~/crane_ws/ops
./run_traj_log.sh play --no-arm ~/csv_profiles/pulse.csv
./run_traj_log.sh show
```

`show` should report **Lines: 6** or more (header + PROFILE_START + 4 waypoints + PROFILE_DONE).

**Windows PC — download:**

```powershell
mkdir $HOME\Downloads\jetson_logs
scp sanjay@192.168.0.101:~/payload_logs/*_traj_cmd.csv $HOME\Downloads\jetson_logs\
```

---

## Path B — Full crane run (USB hub connected)

Plug in ClearPath USB so this works:

```bash
ls /dev/ttyXRUSB*
```

**Terminal 1:**

```bash
source /opt/ros/humble/setup.bash
source ~/crane_ws/install/setup.bash
cd ~/crane_ws/ops
./run_traj_log.sh launch
```

Gantry must show **CoreXY Gantry Controller** without `No SC4-HUB found`.

**Terminal 2:**

```bash
source /opt/ros/humble/setup.bash
source ~/crane_ws/install/setup.bash
cd ~/crane_ws/ops
./run_traj_log.sh play ~/csv_profiles/pulse.csv
./run_traj_log.sh enable
```

**Sensor homing:** With `auto_home_before_run: true`, `./run_traj_log.sh enable` homes motor A then B (limit switches) **before** `MOTION_START`. Watch:

```bash
ros2 topic echo /gantry/state --field homing_status
```

Clear workspace; configure homing in ClearView first (`ops/HANDOFF.md`).

```bash
./run_traj_log.sh show
```

---

## What went wrong in your session

| Issue | Cause |
|-------|--------|
| `Package gantry_control not found` (T2) | Did not `source ~/clearpath_ws/install/setup.bash` |
| `AMENT_TRACE_SETUP_FILES: unbound variable` | Old script used `set -u`; use updated `run_traj_log.sh` |
| `No SC4-HUB found` (T1) | Crane USB not connected — gantry died; traj_player could not arm TRAJ |
| `0` byte `*_traj_cmd.csv` | traj_player never ran successfully; no `/traj_cmd` messages |
| Logger “No active pose streams” | Normal for traj-only; ignore if you only need `*_traj_cmd.csv` |

**`traj_player`** lives in **`gantry_control`** (`crane_ws`).  
**`experiment_logger`** / **`payload_perception`** are separate packages in **`crane_ws`**.  
CSV profiles stay in **`~/csv_profiles/`**.

---

## Optional: vision logging

For `logger_sync_pose_<time>.csv` (pose), use full launch with camera.
Jetson = **192.168.0.101** (SSH, web stream). OAK = **192.168.0.153** (`oak_ip`).

```bash
ros2 launch gantry_control gantry.launch.py oak_ip:=192.168.0.153 start_encoder:=false
```

Web preview: http://192.168.0.101:8080/

Then run `traj_player` + `enable` in Terminal 2 as in Path B.
