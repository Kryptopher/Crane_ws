# JOG joystick (Logitech Dual Action)

## If the cart does not move in JOG mode

1. Check `/joy` is publishing (~10–50 Hz):
   ```bash
   ros2 topic hz /joy
   ```
   If it says "does not appear to be published", restart launch (see below).

2. Launch must set `SDL_JOYSTICK_DEVICE` (fixed in `gantry.launch.py`). Manual workaround:
   ```bash
   export SDL_JOYSTICK_DEVICE=/dev/input/js0
   ros2 run joy joy_node --ros-args -p device_id:=0 -p autorepeat_rate:=50.0
   ```
   Look for: `Opened joystick: Logitech Dual Action`

## Controls (Logitech Dual Action)

| Action | Control |
|--------|---------|
| **Enable jog** | Hold **L1** (left shoulder, *PinkieBtn* — button index 4) |
| **Move** | **Left stick** while L1 held (right → +X, up → +Y) |
| **E-stop** | **R1** (right shoulder, button 5) — avoid while testing |
| Speed preset | **Y** / **A** buttons (cycle jog speed) |

The code maps **Xbox-style** indices; on this pad, **L1 = button 4** matches `buttons[4]`.

## JOG setup

```bash
ros2 run gantry_control gantry_cli.py enable
ros2 run gantry_control gantry_cli.py home
ros2 run gantry_control gantry_cli.py jog 0
```

## Position verify

```bash
ros2 run gantry_control jog_position_verify.py
```
