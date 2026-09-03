#ifndef GANTRY_CONTROL__GANTRY_CONTROLLER_HPP_
#define GANTRY_CONTROL__GANTRY_CONTROLLER_HPP_

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/transform_broadcaster.h"

// sFoundation headers
#include "pubSysCls.h"
#include "pubCpmAdvAPI.h"
#include "pubMotion.h"

// Custom messages/services (generated)
#include "gantry_control/msg/gantry_state.hpp"
#include "gantry_control/msg/traj_cmd.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "gantry_control/srv/set_mode.hpp"
#include "gantry_control/srv/move_to.hpp"
#include "gantry_control/srv/execute_timed_profile.hpp"

namespace gantry_control
{

// ============================================================================
// CSV profile entry
// ============================================================================
struct CsvEntry
{
  double time_s;
  double vx_mm_s;
  double vy_mm_s;
};

// ============================================================================
// Operating modes
// ============================================================================
enum class Mode
{
  IDLE,
  HOMING,
  JOG,
  CSV,
  TRAJ,
  MISSION,
  ZV_JOG
};

enum class HomingMode
{
  SensorInputs,
  TeknicHoming
};

enum class HomingPhase
{
  Idle,
  Validate,
  SensorCheck,
  ReleaseHomeSwitches,
  SeekBack,
  SeekLeft,
  StartA,
  WaitA,
  StartB,
  WaitB,
  Finalize,
  Failed
};

enum class PendingRunKind
{
  None,
  ManualHome,
  TrajBuffered,
  TrajRealtime,
  Csv,
  Mission
};

inline std::string mode_to_string(Mode m)
{
  switch (m) {
    case Mode::IDLE:    return "IDLE";
    case Mode::HOMING:  return "HOMING";
    case Mode::JOG:     return "JOG";
    case Mode::CSV:     return "CSV";
    case Mode::TRAJ:    return "TRAJ";
    case Mode::MISSION: return "MISSION";
    case Mode::ZV_JOG: return "ZV_JOG";
    default:            return "UNKNOWN";
  }
}

// ============================================================================
// GantryController — main ROS2 node
// ============================================================================
class GantryController : public rclcpp::Node
{
public:
  explicit GantryController(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~GantryController() override;
  /** SIGINT / launch shutdown: zero motion, disable, close Teknic port (no E-stop latch). */
  void shutdown_for_exit();

private:
  // ──────────────────────────────────────────────────────────────────────────
  // ROS2 interfaces
  // ──────────────────────────────────────────────────────────────────────────

  // Publishers
  rclcpp::Publisher<gantry_control::msg::GantryState>::SharedPtr state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<gantry_control::msg::TrajCmd>::SharedPtr traj_cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr read_timing_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr traj_latency_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr home_sensors_pub_;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Subscription<gantry_control::msg::TrajCmd>::SharedPtr traj_cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr payload_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr pose_sync_hz_sub_;

  // Services
  rclcpp::Service<gantry_control::srv::SetMode>::SharedPtr set_mode_srv_;
  rclcpp::Service<gantry_control::srv::MoveTo>::SharedPtr move_to_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr estop_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_estop_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr enable_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr disable_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr force_home_srv_;
  rclcpp::Service<gantry_control::srv::ExecuteTimedProfile>::SharedPtr
    execute_timed_profile_srv_;

  // Timers
  rclcpp::TimerBase::SharedPtr control_timer_;   // 100Hz main loop
  rclcpp::TimerBase::SharedPtr state_timer_;      // 50Hz state publisher
  rclcpp::TimerBase::SharedPtr home_sensors_timer_;  // 10Hz home-input diagnostics

  // ──────────────────────────────────────────────────────────────────────────
  // Callbacks
  // ──────────────────────────────────────────────────────────────────────────
  void control_loop();
  void publish_state();
  /** 10 Hz dump of the raw SC4-hub home inputs + derived active flags, for
   *  verifying home-sensor wiring/polarity. Topic: /gantry/home_sensors. */
  void publish_home_sensors();
  void publish_gantry_odometry(const rclcpp::Time & stamp);
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg);
  void traj_cmd_callback(const gantry_control::msg::TrajCmd::SharedPtr msg);
  void payload_pose_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void pose_sync_hz_callback(const std_msgs::msg::Float64::SharedPtr msg);
  void reset_state_timer(double hz);
  void publish_state_if_due();
  void trip_payload_position_limit(const char * axis_label, double value_m);
  void set_mode_callback(
    const std::shared_ptr<gantry_control::srv::SetMode::Request> request,
    std::shared_ptr<gantry_control::srv::SetMode::Response> response);
  void move_to_callback(
    const std::shared_ptr<gantry_control::srv::MoveTo::Request> request,
    std::shared_ptr<gantry_control::srv::MoveTo::Response> response);
  void estop_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void clear_estop_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void enable_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void disable_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  /** Force the current gantry position to be the software home, with no seek. */
  void force_home_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void execute_timed_profile_callback(
    const std::shared_ptr<gantry_control::srv::ExecuteTimedProfile::Request> request,
    std::shared_ptr<gantry_control::srv::ExecuteTimedProfile::Response> response);

  // ──────────────────────────────────────────────────────────────────────────
  // Mode executors (called from control_loop)
  // ──────────────────────────────────────────────────────────────────────────
  void execute_idle();
  void execute_homing();
  void execute_homing_sensor_inputs();
  void execute_homing_teknic();
  void begin_homing(PendingRunKind pending, Mode resume_mode);
  /** Raw hub input level (true = electrically asserted), no polarity applied. */
  bool read_hub_input_raw(size_t node_index, bool input_b);
  bool read_home_hub_input(size_t node_index, bool input_b, bool active_high);
  bool home_left_sensor_active();
  bool home_back_sensor_active();
  bool hardware_estop_sensor_active();
  void monitor_hardware_estop();
  bool home_sensors_both_active_debounced();
  void reset_home_input_debounce();
  void fail_homing(const std::string & reason);
  void finalize_homing_success();
  /** Capture current motor positions as the home reference and zero the cart
   *  pose. Shared by sensor homing and the /gantry/force_home override.
   *  Returns false if the motor encoders could not be read. */
  bool set_home_reference_here();
  void dispatch_pending_run();
  void do_start_traj_execution();
  void do_start_traj_realtime(bool reset_stream_command = true);
  void do_start_csv_run();
  void execute_jog();
  void execute_csv();
  void execute_traj();
  void execute_mission();
  void start_traj_execution();
  void start_traj_realtime(bool reset_stream_command = true);
  bool check_motor_homing_valid();
  void publish_motion_start();
  void publish_traj_playback(double profile_time_s);
  void publish_traj_playback_realtime(double profile_time_s, double vx_mm_s, double vy_mm_s);
  void clear_traj_profile();
  bool send_cartesian_velocity(double vx_ms, double vy_ms);
  bool send_traj_cartesian_velocity(double vx_ms, double vy_ms, bool force = false);
  void reset_traj_write_cache();
  bool check_workspace_position();
  void clamp_velocity_to_workspace(double & vx_ms, double & vy_ms);
  void trip_workspace_limit(const char * axis_label, double value_m);
  void execute_zv_jog();

  // ──────────────────────────────────────────────────────────────────────────
  // sFoundation hardware interface
  // ──────────────────────────────────────────────────────────────────────────
  bool init_hardware();
  void shutdown_hardware();
  bool enable_motors();
  void disable_motors();
  void emergency_stop();
  void clear_emergency_stop();

  // Read encoder states into local variables
  void read_encoders();

  // FK cart pose before homing_cart_bias (for homing zero)
  void forward_cartesian_unbiased(double & x_m, double & y_m);

  // Synchronized velocity command (both motors start simultaneously)
  void send_velocity_synced(double vel_a_rpm, double vel_b_rpm);

  // Synchronized position command (both motors start simultaneously)
  void send_position_synced(int32_t pos_a_counts, int32_t pos_b_counts);

  // Cap Teknic VelLimit/AccLimit for MISSION position moves (restored after move / mode change)
  void apply_mission_move_limits(double vx_ms, double vy_ms);
  void restore_motor_limits();

  // Immediate velocity command (single motor, for special cases)
  void send_velocity_immediate(size_t motor_idx, double vel_counts_per_sec);

  // Check if position move is done on both motors
  bool is_move_done();

  // ──────────────────────────────────────────────────────────────────────────
  // CoreXY kinematics
  // ──────────────────────────────────────────────────────────────────────────

  // Forward: motor positions (rad) → Cartesian (m)
  void forward_kinematics(double motor_a_rad, double motor_b_rad,
                          double & x_m, double & y_m);

  // Forward: motor velocities (rad/s) → Cartesian velocities (m/s)
  void forward_velocity(double motor_a_rads, double motor_b_rads,
                        double & vx_ms, double & vy_ms);

  // Inverse: Cartesian velocity (m/s) → motor RPM
  void inverse_velocity(double vx_ms, double vy_ms,
                        double & motor_a_rpm, double & motor_b_rpm);

  // Inverse: Cartesian position (m) → motor counts (relative to home)
  void inverse_position(double x_m, double y_m,
                        int32_t & motor_a_counts, int32_t & motor_b_counts);

  // ──────────────────────────────────────────────────────────────────────────
  // CSV profile helpers
  // ──────────────────────────────────────────────────────────────────────────
  bool load_csv_profile(const std::string & path);
  std::vector<CsvEntry> csv_profile_;
  size_t csv_index_;
  std::chrono::steady_clock::time_point csv_start_time_;

  // ──────────────────────────────────────────────────────────────────────────
  // Hardware objects
  // ──────────────────────────────────────────────────────────────────────────
  sFnd::SysManager * sys_mgr_;
  sFnd::IPort * port_;
  sFnd::INode * node_a_;   // Motor A
  sFnd::INode * node_b_;   // Motor B
  int teknic_baud_rate_;

  // ──────────────────────────────────────────────────────────────────────────
  // System state (protected by mutex for service callbacks)
  // ──────────────────────────────────────────────────────────────────────────
  std::mutex state_mutex_;

  Mode current_mode_;
  bool hardware_initialized_;
  bool motors_enabled_;
  bool homed_;
  bool homing_active_;
  std::string homing_status_;
  HomingPhase homing_phase_;
  PendingRunKind pending_run_kind_;
  Mode mode_before_homing_;
  bool homing_a_active_seen_;
  bool homing_b_active_seen_;
  std::chrono::steady_clock::time_point homing_deadline_;
  bool auto_home_before_run_;
  HomingMode homing_mode_;
  double homing_timeout_s_;
  bool homing_sequential_;
  double homing_seek_vel_ms_;
  double homing_seek_vx_ms_;
  double homing_seek_vy_ms_;
  size_t home_left_node_;
  size_t home_back_node_;
  bool home_left_input_b_;
  bool home_back_input_b_;
  bool home_back_input_active_high_;
  bool home_left_input_active_high_;
  int homing_input_debounce_ticks_;
  int homing_left_debounce_count_;
  int homing_back_debounce_count_;
  bool hardware_estop_enable_;
  size_t hardware_estop_node_;
  bool hardware_estop_input_b_;
  bool hardware_estop_active_high_;
  int hardware_estop_debounce_ticks_;
  int hardware_estop_debounce_count_;
  bool hardware_estop_input_active_;
  bool publish_odom_;
  bool publish_odom_tf_;
  std::string odom_frame_id_;
  std::string odom_child_frame_id_;
  bool estop_active_;
  bool move_in_progress_;

  // Encoder state (updated in control loop)
  double motor_a_pos_rad_;      // Motor A position (radians)
  double motor_a_vel_rads_;     // Motor A velocity (rad/s)
  double motor_b_pos_rad_;      // Motor B position (radians)
  double motor_b_vel_rads_;     // Motor B velocity (rad/s)

  // Cartesian state (computed from encoder state)
  double cart_x_m_;             // Cartesian X position (meters)
  double cart_y_m_;             // Cartesian Y position (meters)
  double cart_vx_ms_;           // Cartesian X velocity (m/s)
  double cart_vy_ms_;           // Cartesian Y velocity (m/s)
  double cart_vx_measured_ms_;
  double cart_vy_measured_ms_;
  double cart_vx_from_position_ms_;
  double cart_vy_from_position_ms_;
  double last_cart_x_for_velocity_m_;
  double last_cart_y_for_velocity_m_;
  double position_velocity_alpha_;
  bool position_velocity_initialized_;
  std::chrono::steady_clock::time_point last_position_velocity_time_;
  std::string cart_velocity_source_;
  int motor_velocity_read_every_n_;
  int motor_velocity_read_counter_;

  // Home offsets (encoder counts at home position)
  int32_t home_offset_a_;
  int32_t home_offset_b_;
  /** FK bias so reported cart is 0,0 at home sensors (physical pose may differ). */
  double homing_cart_bias_x_;
  double homing_cart_bias_y_;

  // ──────────────────────────────────────────────────────────────────────────
  // Jog state
  // ──────────────────────────────────────────────────────────────────────────
  double jog_vx_;               // Current jog velocity command X (m/s)
  double jog_vy_;               // Current jog velocity command Y (m/s)
  int jog_speed_preset_;        // 0=slow, 1=medium, 2=fast
  static constexpr double JOG_SPEEDS[] = {0.100, 0.300, 0.500};  // m/s
  bool joy_enable_held_;        // LB button state
  double joy_axis_sign_x_;      // stick → cart +X (lab: origin bottom-left, right = +X)
  double joy_axis_sign_y_;      // stick → cart +Y (lab: up = +Y)

  // ──────────────────────────────────────────────────────────────────────────
  // ZVD Jog state (robust 3-impulse Zero-Vibration-Derivative input shaper)
  // ──────────────────────────────────────────────────────────────────────────
  enum class ZvState { IDLE, RAMP_UP_1, RAMP_UP_2, FULL_SPEED, RAMP_DOWN_1, RAMP_DOWN_2 };
  ZvState zv_state_;
  std::chrono::steady_clock::time_point zv_transition_time_;
  double zv_T_;                 // Impulse spacing = pi / omega_d (seconds)
  double zv_zeta_;              // Damping ratio used for ZVD amplitude computation
  double zv_dir_x_;             // Direction: -1, 0, or +1
  double zv_dir_y_;             // Direction: -1, 0, or +1
  bool zv_button_held_;         // Stick+LB held state

  // ──────────────────────────────────────────────────────────────────────────
  // Mission state
  // ──────────────────────────────────────────────────────────────────────────
  double mission_target_x_;
  double mission_target_y_;
  double mission_move_vel_ms_;
  double mission_move_accel_ms2_;
  // Runtime retuning of the move limits above (dashboard experiment runs
  // command a specific velocity per move via /gantry_controller/set_parameters).
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // ──────────────────────────────────────────────────────────────────────────
  // TRAJ state (/traj_cmd from traj_player.py)
  // ──────────────────────────────────────────────────────────────────────────
  std::vector<CsvEntry> traj_profile_;
  size_t traj_index_;
  std::chrono::steady_clock::time_point traj_start_time_;
  bool traj_profile_loaded_;
  bool traj_running_;
  bool traj_abort_;
  bool traj_realtime_enable_;
  bool traj_realtime_active_;
  bool timed_profile_armed_;
  bool precise_profile_timing_;
  std::chrono::steady_clock::time_point timed_profile_start_time_;
  double timed_profile_start_ros_s_;
  double stream_vx_mm_s_;
  double stream_vy_mm_s_;
  double traj_stream_timeout_s_;
  bool traj_write_on_change_only_;
  double traj_write_keepalive_s_;
  double traj_write_deadband_mm_s_;
  double last_sent_traj_vx_ms_;
  double last_sent_traj_vy_ms_;
  bool last_sent_traj_velocity_valid_;
  std::chrono::steady_clock::time_point last_traj_write_time_;
  uint64_t traj_stream_seq_{0};
  uint64_t traj_applied_seq_{0};
  bool traj_received_velocity_valid_{false};
  double traj_received_vx_mm_s_{0.0};
  double traj_received_vy_mm_s_{0.0};
  double traj_source_stamp_s_{0.0};
  double traj_rx_stamp_s_{0.0};
  double traj_apply_begin_stamp_s_{0.0};
  double traj_apply_done_stamp_s_{0.0};
  double traj_applied_vx_mm_s_{0.0};
  double traj_applied_vy_mm_s_{0.0};
  double encoder_read_stamp_s_{0.0};

  // Gantry Cartesian workspace safety (motor encoders)
  double workspace_limit_m_;
  bool workspace_limit_enable_;
  bool workspace_limit_tripped_;

  // Payload vision safety (/payload/pose) — optional, off by default
  double payload_position_limit_m_;
  bool payload_limit_tripped_;
  bool payload_limit_monitor_enable_;

  // ──────────────────────────────────────────────────────────────────────────
  // Watchdog
  // ──────────────────────────────────────────────────────────────────────────
  std::chrono::steady_clock::time_point last_cmd_time_;
  static constexpr double WATCHDOG_TIMEOUT_S = 0.5;

  // ──────────────────────────────────────────────────────────────────────────
  // Timing diagnostics
  // ──────────────────────────────────────────────────────────────────────────
  double last_read_ms_;
  double last_write_ms_;
  double last_pos_a_read_ms_;
  double last_pos_b_read_ms_;
  double last_vel_a_read_ms_;
  double last_vel_b_read_ms_;
  double last_read_math_ms_;
  uint32_t error_count_;
  double stack_pose_publish_hz_;
  bool stack_pose_sync_adaptive_;
  double measured_state_rate_hz_;
  std::chrono::steady_clock::time_point last_state_publish_time_;

  // ──────────────────────────────────────────────────────────────────────────
  // Physical constants
  // ──────────────────────────────────────────────────────────────────────────
  static constexpr double PULLEY_RADIUS_M = 0.01989;
  static constexpr double GEAR_RATIO = 5.0;
  static constexpr double COUNTS_PER_REV = 800.0;
  static constexpr double R_EFF = PULLEY_RADIUS_M / GEAR_RATIO;  // effective radius
  static constexpr double WORKSPACE_M = 1.22;     // 4ft
  static constexpr double MAX_VEL_MS = 1.0;       // m/s absolute max
  static constexpr double MAX_MOTOR_RPM = 2800.0;
  static constexpr double MAX_ACCEL_RPM_S = 100000.0;  // High for step response in CSV mode
  static constexpr double POSITION_TOLERANCE_M = 0.002;  // 2mm

  // Trigger group for synchronized moves
  static constexpr size_t SYNC_TRIGGER_GROUP = 1;
};

}  // namespace gantry_control

#endif  // GANTRY_CONTROL__GANTRY_CONTROLLER_HPP_
