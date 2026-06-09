# Trajectory command for TRAJ mode (published by traj_player.py)
std_msgs/Header header

float64 time_s       # knot time (seconds from profile start)
float64 x            # integrated X (meters, for logging)
float64 y            # integrated Y (meters, for logging)

float64 vx_mm_s      # Cartesian velocity for this segment (mm/s, crane frame)
float64 vy_mm_s

float64 motor_a_pos_rad   # ClearPath motor A encoder (PLAYBACK only)
float64 motor_b_pos_rad   # ClearPath motor B encoder (PLAYBACK only)
float64 motor_a_vel_rad_s
float64 motor_b_vel_rad_s

# command: 0=WAYPOINT, 1=PROFILE_DONE, 2=ABORT, 3=PROFILE_START, 4=MOTION_START, 5=PLAYBACK, 6=STREAM
uint8 command

uint8 WAYPOINT=0
uint8 PROFILE_DONE=1
uint8 ABORT=2
uint8 PROFILE_START=3
uint8 MOTION_START=4    # gantry publishes at playback t=0 (motors moving)
uint8 PLAYBACK=5        # gantry 100Hz: traj + motor encoders (PosnMeasured read)
uint8 STREAM=6          # real-time velocity command (vx_mm_s, vy_mm_s) while TRAJ active
