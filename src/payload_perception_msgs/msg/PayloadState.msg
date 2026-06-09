# Payload pose from AprilTag vision (+ gantry frame when calibrated).
std_msgs/Header header

float64 motion_time_sec   # seconds since MOTION_START (experiment time)

# Per-tag pair (legacy names): camera optical frame, session-relative (m)
#   x1 = tag A camera X,  z1 = tag A camera Z (horizontal plane)
#   x2, z2 = second tag — NOT gantry axes
float64 x1
float64 z1
float64 x2
float64 z2

float64 vx1               # tag velocities in camera axes (m/s)
float64 vz1
float64 vx2
float64 vz2

# Payload box center, camera frame (m), session-relative
#   cam_x, cam_z = horizontal in view; cam_y = vertical (rope axis, usually small motion)
float64 cam_x
float64 cam_y
float64 cam_z

# Payload box center, gantry frame (m), session-relative (tripod camera: R_mount only)
#   gantry_x, gantry_y = cart plane; gantry_z = out-of-plane swing
float64 gantry_x
float64 gantry_y
float64 gantry_z

float64 v_gantry_x
float64 v_gantry_y
float64 v_gantry_z

bool valid
bool interpolated
string frame_id           # "gantry" when gantry_* filled, else "camera"
