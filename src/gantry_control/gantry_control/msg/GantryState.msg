# Gantry state feedback message
std_msgs/Header header

# Cartesian position (meters, relative to home)
float64 x
float64 y

# Cartesian velocity (m/s)
float64 vx
float64 vy

# Motor states (radians, rad/s)
float64 motor_a_position
float64 motor_a_velocity
float64 motor_b_position
float64 motor_b_velocity

# System status
string mode          # IDLE, HOMING, JOG, CSV, TRAJ, MISSION, ZV_JOG
bool homed           # True if homing has been completed
bool homing_active   # True while sensor homing sequence is running
string homing_status # validate, motor_a, motor_b, done, failed, ...
bool move_done       # True if current position move is complete
bool enabled         # True if motors are enabled
bool estop           # True if emergency stop is active

# Diagnostics
float64 loop_rate_hz
float64 read_time_ms
float64 write_time_ms
uint32 error_count
