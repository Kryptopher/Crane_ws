# Payload frame conventions (camera-only)

## Lab setup: OAK on tripod (not on cart)

The camera is **fixed in the room**. The gantry cart moves underneath; the camera does **not**
ride on the cart. You do **not** need cart position or `payload_gantry_frame` for this project.

Only one step is required:

| Step | What | Where |
|------|------|--------|
| **Axis conversion** | Camera (X,Z horizontal, Y vertical) → gantry (X,Y cart plane, Z swing) | **`payload_tracker`** |

Fixed rotation `R_mount` from tripod aim — calibrate once with `yaw` / `pitch` / `roll` / signs.

## Front-facing tripod (this lab)

Camera looks at the payload **from the front** (fixed on tripod). AprilTag pose uses the
usual optical frame (OpenCV / OAK):

| What you do | What the camera sees | Tracker field |
|-------------|----------------------|---------------|
| Payload swings (pendulum) | Mostly **left/right** in the image | **`cam_x`** (and a little `cam_y`) |
| Gantry **X+** (cart along X) | Tag moves **left** in the frame | **`cam_x`** decreases (opposite sign to gantry X) |
| Gantry **Y+** (cart toward camera) | Tag gets **closer** | **`cam_z`** decreases (depth / range, not `cam_y`) |
| Vertical in the image | Up/down on the sensor | **`cam_y`** |

So cart travel in the horizontal plane shows up as **`cam_x` + `cam_z`**, not as cart encoders
in the tracker. That is **correct** for a fixed camera — the tag really moves in the image.

`gantry_x` / `gantry_y` / `gantry_z` are the same motion expressed in crane axes after
`R_mount` and sign flips. Calibration goal:

- Jog **X only** → mostly **`gantry_x`** (via `cam_x`, sign often `--gantry-sign-x -1` if X+ looks left)
- Jog **Y only** → mostly **`gantry_y`** (via **`cam_z`**, depth)
- **Swing only** → mostly **`gantry_z`** (today may still look like `cam_x` until yaw/pitch/roll are tuned)

System ID swing uses **`gantry_z`** on `/payload/state` after calibration; until then you can
run `adaptive_id_player` with `--swing-field cam_x` to match what you see.

## `payload_tracker` → `/payload/state`

- `cam_x`, `cam_y`, `cam_z` — box center, **camera/tripod** frame (session-relative m)
- `gantry_x`, `gantry_y`, `gantry_z` — same center, **gantry** axes (`R_mount @ cam`)
- `x1`, `z1` — legacy per-tag samples in camera horizontal plane
- `frame_id: gantry` when `--ros` (default)

No `/gantry/state` subscription in the tracker.

## `payload_gantry_frame` — do not use here

That node subtracts cart travel for a **camera mounted on the moving cart**. Ignore it for
tripod-mounted OAK (`start_gantry_frame:=false`, the default).

## Consumers

| Tool | Topic / fields |
|------|----------------|
| Mission planner plot | `gantry_x` vs `gantry_z` on `/payload/state` |
| adaptive_id_player | `--payload-topic /payload/state --payload-type payloadstate` and field `gantry_z`, or legacy array index on `gantry_z` |

Example (swing on gantry Z):

```bash
ros2 run gantry_control adaptive_id_player.py \
  --payload-topic /payload/state \
  --payload-type payloadstate
```

(Use `--pose-x-index` only for `float64multiarray` legacy layout.)

## Plot smoothing

Pipeline (tripod camera):

1. **World pose** median+EMA before session zero (`world_smoother`)
2. **Display overlay** EMA (`--pos-alpha`, launch default `0.40`)
3. **ROS publish:** session-relative center → `_cam_smoother` → mount rotation → `_gantry_smoother`
4. Dashboard vision chart: light display EMA (`alpha≈0.42`) + trail history

Spike clamp: `--publish-max-step-m` (launch default `0.008` m per frame).

**Reset session zero:** dashboard **Reset vision origin** or `ros2 topic pub --once /payload/reset_origin std_msgs/msg/Empty {}`

Tuning (launch args on `gantry.launch.py`): still noisy → lower `tracker_publish_alpha`; sluggish trail → raise `tracker_pos_alpha` slightly.

## Bench checklist (tripod)

1. Launch vision-only: `mission_planner.launch.py start_gantry:=false start_encoder:=false`
2. Hold payload at rest 3 s → vision chart dot near **session 0**; note `|r|` in meta line
3. Single swing and return → dot should return within ~2–5 cm of start; use **Reset vision origin** between trials
4. Toggle dashboard **Axes: camera X–Z** if gantry plot looks wrong (mount not calibrated yet)
5. Jog cart **X only** → mostly `gantry_x`; **Y only** → mostly `gantry_y` (via `cam_z`); **swing only** → mostly `gantry_z`

## Calibration

Pass launch args (or edit tracker CLI):

```bash
ros2 launch gantry_control mission_planner.launch.py \
  gantry_yaw_deg:=0.0 gantry_sign_x:=-1.0 gantry_sign_y:=1.0 gantry_sign_z:=1.0
```

Jog or swing manually until:

- Pendulum swing → **gantry_z** changes
- **gantry_y** stays small (rope direction)
- **gantry_x** only changes when you move the cart in X (real motion, not coupling)

Saved defaults: edit `src/gantry_control/gantry_control/config/payload_mount.yaml` (loaded by `gantry.launch.py` / `mission_planner.launch.py`).

## Metric scale (dimensional accuracy)

| Check | How | Pass |
|-------|-----|------|
| Tag size | Measure printed AprilTag **outer black square** edge-to-edge (m). Set `marker_size` / launch `marker_size:=` to that value. | Depth error scales with wrong tag size |
| Depth step | Move payload **0.20 m** toward/away from camera (measure with tape). Watch `cam_z` delta on `/payload/state`. | \|Δcam_z\| within **±15%** of 0.20 m |
| Cart jog | After mount cal, jog cart **0.10 m** in X only. | \|Δgantry_x\| within **±1.5 cm** of 0.10 m |
| Box model | Code constants in `payload_tracker.py`: box **13.50 × 12.00 in** outer, tag spacing **3.75 in**. Match physical box or update `_BOX_HALF_*`. | Face geometry stable when box rotates |
| Accuracy vs speed | `tracker_quad_decimate:=2.0` (default). Use `2.5` only for debug FPS, not metrology. | — |

OAK intrinsics come from the device; wrong `marker_size` is the most common scale error on a tripod rig.
