# crane_ws

Unified ROS 2 workspace for the payload crane experiment stack.

## Packages

| Package | Role |
|---------|------|
| `gantry_control` | **Control** — ClearPath gantry, TRAJ, `TrajCmd` msgs, `traj_player`, launches |
| `payload_perception` | **Perception** — OAK + AprilTags (`payload_tracker`), GPIO encoders (`encoder_node`) |
| `experiment_logger` | **Logging** — CSV logger for `/traj_cmd` and synced pose streams |

## Shell setup (every terminal)

```bash
source /opt/ros/humble/setup.bash
source ~/crane_ws/install/setup.bash
```

## Build

```bash
cd ~/crane_ws
colcon build
source install/setup.bash
```

## Operator scripts

See **`ops/HANDOFF.md`** (main handoff), `ops/PROJECT_HANDOFF.txt` (short paste), `ops/TRAJ_RUNBOOK.md`, `ops/PAYLOAD_FRAMES.md`.

## Data (outside workspace)

- `~/csv_profiles/` — velocity trajectories
- `~/payload_logs/` — experiment CSV output

## Network

- Jetson: `192.168.0.101` (SSH, web stream `:8080`)
- OAK camera: `192.168.0.153` (`oak_ip` launch arg)

## Legacy paths

- `~/clearpath_ws` — superseded by `crane_ws` (keep until you delete)
- `~/payload_tracker` — symlink to `crane_ws/ops` after migration
