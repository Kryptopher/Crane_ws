#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include "gantry_control/gantry_controller.hpp"
#include <functional>
#include <geometry_msgs/msg/transform_stamped.hpp>

using namespace std::chrono_literals;
using std::placeholders::_1;
using std::placeholders::_2;

namespace gantry_control
{

// Static member definition
constexpr double GantryController::JOG_SPEEDS[];

// ============================================================================
// CONSTRUCTOR
// ============================================================================
GantryController::GantryController(const rclcpp::NodeOptions & options)
: Node("gantry_controller", options),
  sys_mgr_(nullptr), port_(nullptr), node_a_(nullptr), node_b_(nullptr),
  
  current_mode_(Mode::IDLE),
  hardware_initialized_(false), motors_enabled_(false),
  homed_(false), homing_active_(false), homing_status_("idle"),
  homing_phase_(HomingPhase::Idle), pending_run_kind_(PendingRunKind::None),
  mode_before_homing_(Mode::IDLE),
  homing_a_active_seen_(false), homing_b_active_seen_(false),
  auto_home_before_run_(true), homing_mode_(HomingMode::SensorInputs),
  homing_timeout_s_(30.0), homing_sequential_(true), homing_seek_vel_ms_(0.05),
  homing_seek_vx_ms_(-0.05), homing_seek_vy_ms_(-0.05),
  home_back_node_(0), home_left_node_(1),
  home_back_input_b_(false), home_left_input_b_(true),
  home_back_input_active_high_(false), home_left_input_active_high_(true),
  homing_input_debounce_ticks_(5),
  homing_left_debounce_count_(0), homing_back_debounce_count_(0),
  publish_odom_(true), publish_odom_tf_(true),
  odom_frame_id_("gantry"), odom_child_frame_id_("gantry_cart"),
  estop_active_(false), move_in_progress_(false),
  motor_a_pos_rad_(0), motor_a_vel_rads_(0),
  motor_b_pos_rad_(0), motor_b_vel_rads_(0),
  cart_x_m_(0), cart_y_m_(0), cart_vx_ms_(0), cart_vy_ms_(0),
  home_offset_a_(0), home_offset_b_(0),
  homing_cart_bias_x_(0), homing_cart_bias_y_(0),
  jog_vx_(0), jog_vy_(0), jog_speed_preset_(0),
  joy_enable_held_(false),
  zv_state_(ZvState::IDLE), zv_T_(1.0), zv_A_(0.300),
  zv_dir_x_(0), zv_dir_y_(0), zv_button_held_(false),
  mission_target_x_(0), mission_target_y_(0),
  mission_move_vel_ms_(0.10), mission_move_accel_ms2_(0.20),
  traj_index_(0), traj_profile_loaded_(false),
  traj_running_(false), traj_abort_(false),
  traj_realtime_enable_(true), traj_realtime_active_(false),
  stream_vx_mm_s_(0.0), stream_vy_mm_s_(0.0),
  traj_stream_timeout_s_(0.5),
  workspace_limit_m_(1.0), workspace_limit_enable_(true),
  workspace_limit_tripped_(false),
  payload_position_limit_m_(1.0), payload_limit_tripped_(false),
  payload_limit_monitor_enable_(false),
  last_read_ms_(0), last_write_ms_(0), error_count_(0),
  stack_pose_publish_hz_(50.0), stack_pose_sync_adaptive_(true),
  measured_state_rate_hz_(0.0)
{
  RCLCPP_INFO(get_logger(), "═══════════════════════════════════════════");
  RCLCPP_INFO(get_logger(), "  CoreXY Gantry Controller v2.0");
  RCLCPP_INFO(get_logger(), "  Stein Lab — LSU");
  RCLCPP_INFO(get_logger(), "═══════════════════════════════════════════");

  // ── Publishers ──
  state_pub_ = create_publisher<gantry_control::msg::GantryState>("/gantry/state", 10);
  joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
  odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/gantry/odom", 10);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  traj_cmd_pub_ = create_publisher<gantry_control::msg::TrajCmd>("/traj_cmd", 10);

  // ── Subscribers ──
  joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
    "/joy", 10, std::bind(&GantryController::joy_callback, this, _1));

  traj_cmd_sub_ = create_subscription<gantry_control::msg::TrajCmd>(
    "/traj_cmd", 10, std::bind(&GantryController::traj_cmd_callback, this, _1));

  // ── Services ──
  set_mode_srv_ = create_service<gantry_control::srv::SetMode>(
    "/gantry/set_mode",
    std::bind(&GantryController::set_mode_callback, this, _1, _2));

  move_to_srv_ = create_service<gantry_control::srv::MoveTo>(
    "/gantry/move_to",
    std::bind(&GantryController::move_to_callback, this, _1, _2));

  estop_srv_ = create_service<std_srvs::srv::Trigger>(
    "/gantry/estop",
    std::bind(&GantryController::estop_callback, this, _1, _2));

  clear_estop_srv_ = create_service<std_srvs::srv::Trigger>(
    "/gantry/clear_estop",
    std::bind(&GantryController::clear_estop_callback, this, _1, _2));

  enable_srv_ = create_service<std_srvs::srv::Trigger>(
    "/gantry/enable",
    std::bind(&GantryController::enable_callback, this, _1, _2));

  disable_srv_ = create_service<std_srvs::srv::Trigger>(
    "/gantry/disable",
    std::bind(&GantryController::disable_callback, this, _1, _2));

  // ── Initialize hardware ──
  if (!init_hardware()) {
    RCLCPP_FATAL(get_logger(), "Hardware initialization failed!");
    return;
  }

  // ── Parameters ──
  declare_parameter("zv_T", 1.0);
  declare_parameter("zv_A", 0.300);
  zv_T_ = get_parameter("zv_T").as_double();
  zv_A_ = get_parameter("zv_A").as_double();
  RCLCPP_INFO(get_logger(), "ZV params: T=%.3fs A=%.0fmm/s", zv_T_, zv_A_*1000);

  declare_parameter("workspace_limit_m", 1.0);
  declare_parameter("workspace_limit_enable", true);
  workspace_limit_m_ = get_parameter("workspace_limit_m").as_double();
  workspace_limit_enable_ = get_parameter("workspace_limit_enable").as_bool();
  if (workspace_limit_enable_) {
    RCLCPP_INFO(get_logger(),
                "Workspace limit: |x|, |y| ≤ %.3f m (gantry encoders) → E-STOP",
                workspace_limit_m_);
  }

  declare_parameter("traj_realtime_enable", true);
  declare_parameter("traj_stream_timeout_s", 0.5);
  traj_realtime_enable_ = get_parameter("traj_realtime_enable").as_bool();
  traj_stream_timeout_s_ = get_parameter("traj_stream_timeout_s").as_double();
  RCLCPP_INFO(get_logger(), "TRAJ realtime STREAM: %s",
              traj_realtime_enable_ ? "enabled" : "disabled (buffered only)");

  declare_parameter("homing_mode", "sensor_inputs");
  declare_parameter("auto_home_before_run", true);
  declare_parameter("homing_timeout_s", 30.0);
  declare_parameter("homing_sequential", true);
  declare_parameter("homing_seek_vel_ms", 0.05);
  declare_parameter("homing_seek_vx_ms", -0.05);
  declare_parameter("homing_seek_vy_ms", -0.05);
  declare_parameter("home_left_node", 0);
  declare_parameter("home_left_input", "a");
  declare_parameter("home_back_node", 1);
  declare_parameter("home_back_input", "b");
  declare_parameter("home_back_input_active_high", false);
  declare_parameter("home_left_input_active_high", true);
  declare_parameter("homing_input_debounce_ticks", 5);

  const std::string homing_mode_str = get_parameter("homing_mode").as_string();
  homing_mode_ = (homing_mode_str == "teknic_homing")
    ? HomingMode::TeknicHoming : HomingMode::SensorInputs;
  auto_home_before_run_ = get_parameter("auto_home_before_run").as_bool();
  homing_timeout_s_ = get_parameter("homing_timeout_s").as_double();
  homing_sequential_ = get_parameter("homing_sequential").as_bool();
  homing_seek_vel_ms_ = get_parameter("homing_seek_vel_ms").as_double();
  homing_seek_vx_ms_ = get_parameter("homing_seek_vx_ms").as_double();
  homing_seek_vy_ms_ = get_parameter("homing_seek_vy_ms").as_double();
  home_left_node_ = static_cast<size_t>(get_parameter("home_left_node").as_int());
  home_back_node_ = static_cast<size_t>(get_parameter("home_back_node").as_int());
  {
    const std::string in = get_parameter("home_left_input").as_string();
    home_left_input_b_ = (in == "b" || in == "B");
  }
  {
    const std::string in = get_parameter("home_back_input").as_string();
    home_back_input_b_ = (in == "b" || in == "B");
  }
  home_back_input_active_high_ = get_parameter("home_back_input_active_high").as_bool();
  home_left_input_active_high_ = get_parameter("home_left_input_active_high").as_bool();
  homing_input_debounce_ticks_ =
    std::max(1, static_cast<int>(get_parameter("homing_input_debounce_ticks").as_int()));

  RCLCPP_INFO(get_logger(),
              "Homing mode=%s auto_before_run=%s timeout=%.0fs "
              "seek from mission_move_vel_ms (legacy vx/vy params ignored) "
              "left=node%zu in%c back=node%zu in%c back_at=%s left_at=%s",
              homing_mode_ == HomingMode::SensorInputs ? "sensor_inputs" : "teknic_homing",
              auto_home_before_run_ ? "yes" : "no",
              homing_timeout_s_,
              home_left_node_, home_left_input_b_ ? 'B' : 'A',
              home_back_node_, home_back_input_b_ ? 'B' : 'A',
              home_back_input_active_high_ ? "high" : "low",
              home_left_input_active_high_ ? "high" : "low");

  declare_parameter("payload_position_limit_mm", 1000.0);
  declare_parameter("payload_limit_monitor_enable", false);
  payload_position_limit_m_ =
    get_parameter("payload_position_limit_mm").as_double() / 1000.0;
  payload_limit_monitor_enable_ =
    get_parameter("payload_limit_monitor_enable").as_bool();

  declare_parameter("mission_move_vel_ms", 0.10);
  declare_parameter("mission_move_accel_ms2", 0.20);
  mission_move_vel_ms_ = get_parameter("mission_move_vel_ms").as_double();
  mission_move_accel_ms2_ = get_parameter("mission_move_accel_ms2").as_double();
  mission_move_vel_ms_ = std::clamp(mission_move_vel_ms_, 0.01, MAX_VEL_MS);
  mission_move_accel_ms2_ = std::clamp(mission_move_accel_ms2_, 0.05, 2.0);
  RCLCPP_INFO(get_logger(),
              "MISSION move limits: vel=%.3f m/s accel=%.3f m/s²",
              mission_move_vel_ms_, mission_move_accel_ms2_);
  if (payload_limit_monitor_enable_) {
    payload_pose_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/payload/pose", rclcpp::SensorDataQoS(),
      std::bind(&GantryController::payload_pose_callback, this, _1));
    RCLCPP_INFO(get_logger(),
                "Payload limit monitor: |x|,|z| ≤ %.0f mm on /payload/pose → E-STOP",
                payload_position_limit_m_ * 1000.0);
  }

  declare_parameter("publish_odom", true);
  declare_parameter("publish_odom_tf", true);
  declare_parameter("odom_frame_id", "gantry");
  declare_parameter("odom_child_frame_id", "gantry_cart");
  publish_odom_ = get_parameter("publish_odom").as_bool();
  publish_odom_tf_ = get_parameter("publish_odom_tf").as_bool();
  odom_frame_id_ = get_parameter("odom_frame_id").as_string();
  odom_child_frame_id_ = get_parameter("odom_child_frame_id").as_string();
  RCLCPP_INFO(get_logger(),
              "Gantry odometry: /gantry/odom frame=%s child=%s tf=%s",
              odom_frame_id_.c_str(), odom_child_frame_id_.c_str(),
              publish_odom_tf_ ? "yes" : "no");

  declare_parameter("joy_axis_sign_x", -1.0);
  declare_parameter("joy_axis_sign_y", 1.0);
  joy_axis_sign_x_ = get_parameter("joy_axis_sign_x").as_double();
  joy_axis_sign_y_ = get_parameter("joy_axis_sign_y").as_double();
  RCLCPP_INFO(get_logger(),
              "Joystick signs: stick→(+X right, +Y up) = (%.0f, %.0f) × raw axes",
              joy_axis_sign_x_, joy_axis_sign_y_);

  declare_parameter("stack_pose_publish_hz", 50.0);
  declare_parameter("stack_pose_sync_adaptive", true);
  stack_pose_publish_hz_ = get_parameter("stack_pose_publish_hz").as_double();
  stack_pose_publish_hz_ = std::clamp(stack_pose_publish_hz_, 5.0, 200.0);
  stack_pose_sync_adaptive_ = get_parameter("stack_pose_sync_adaptive").as_bool();
  last_state_publish_time_ = std::chrono::steady_clock::now();

  // ── Timers ──
  // 100 Hz control loop (jog, homing, mission)
  control_timer_ = create_wall_timer(10ms, std::bind(&GantryController::control_loop, this));
  if (stack_pose_sync_adaptive_) {
    pose_sync_hz_sub_ = create_subscription<std_msgs::msg::Float64>(
      "/stack/pose_sync_hz", rclcpp::QoS(1).reliable(),
      std::bind(&GantryController::pose_sync_hz_callback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(),
                "State publish: adaptive in control loop via /stack/pose_sync_hz "
                "(bootstrap %.1f Hz, min 20 Hz for UI)",
                stack_pose_publish_hz_);
  } else {
    reset_state_timer(stack_pose_publish_hz_);
    RCLCPP_INFO(get_logger(),
                "State publish: fixed %.1f Hz (/gantry/state, odom, joint_states)",
                stack_pose_publish_hz_);
  }

  last_cmd_time_ = std::chrono::steady_clock::now();

  RCLCPP_INFO(get_logger(), "Controller ready. Call /gantry/enable then /gantry/set_mode");
  RCLCPP_INFO(get_logger(), "Modes: IDLE, HOME, JOG, CSV, TRAJ, MISSION, ZV_JOG");
}

// ============================================================================
// DESTRUCTOR
// ============================================================================
GantryController::~GantryController()
{
  shutdown_for_exit();
}

void GantryController::shutdown_for_exit()
{
  static bool done = false;
  if (done) {
    return;
  }
  done = true;

  RCLCPP_WARN(
    get_logger(),
    "========== SHUTDOWN: zeroing velocity, disabling motors, closing hardware ==========");

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    traj_running_ = false;
    clear_traj_profile();
    move_in_progress_ = false;
    current_mode_ = Mode::IDLE;
    jog_vx_ = 0.0;
    jog_vy_ = 0.0;
    joy_enable_held_ = false;

    if (hardware_initialized_ && motors_enabled_ && !estop_active_) {
      try {
        send_velocity_synced(0.0, 0.0);
      } catch (...) {}
    }
    disable_motors();
    shutdown_hardware();
  }

  RCLCPP_WARN(get_logger(), "========== Gantry controller exit complete ==========");
}

// ============================================================================
// MAIN CONTROL LOOP (100Hz)
// ============================================================================
void GantryController::control_loop()
{
  if (!hardware_initialized_) return;
  auto t0 = std::chrono::steady_clock::now();
  // ── Read motor encoders first (PosnMeasured) for state + TRAJ PLAYBACK log ──
  read_encoders();
  auto t1 = std::chrono::steady_clock::now();
  last_read_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
  if (motors_enabled_ && workspace_limit_enable_) {
    check_workspace_position();
  }
  // ── Execute mode (motor commands) ──
  if (!estop_active_ && motors_enabled_) {
    switch (current_mode_) {
      case Mode::IDLE:    execute_idle();    break;
      case Mode::HOMING:  execute_homing();  break;
      case Mode::JOG:     execute_jog();     break;
      case Mode::CSV:     execute_csv();     break;
      case Mode::TRAJ:    execute_traj();    break;
      case Mode::MISSION: execute_mission(); break;
      case Mode::ZV_JOG: execute_zv_jog(); break;
    }
  }
  auto t2 = std::chrono::steady_clock::now();
  last_write_ms_ = std::chrono::duration<double, std::milli>(t2 - t1).count();

  if (stack_pose_sync_adaptive_) {
    publish_state_if_due();
  }
}
// ============================================================================
// STATE PUBLISHER (stack_pose_publish_hz)
// ============================================================================
void GantryController::publish_state()
{
  if (!hardware_initialized_) return;

  const auto now_tp = std::chrono::steady_clock::now();
  const double dt_s = std::chrono::duration<double>(now_tp - last_state_publish_time_).count();
  last_state_publish_time_ = now_tp;
  if (dt_s > 1e-6) {
    const double inst_hz = 1.0 / dt_s;
    if (measured_state_rate_hz_ < 1.0) {
      measured_state_rate_hz_ = inst_hz;
    } else {
      measured_state_rate_hz_ = 0.9 * measured_state_rate_hz_ + 0.1 * inst_hz;
    }
  }

  std::lock_guard<std::mutex> lock(state_mutex_);

  // ── Gantry state ──
  auto state_msg = gantry_control::msg::GantryState();
  state_msg.header.stamp = now();
  state_msg.x = cart_x_m_;
  state_msg.y = cart_y_m_;
  state_msg.vx = cart_vx_ms_;
  state_msg.vy = cart_vy_ms_;
  state_msg.motor_a_position = motor_a_pos_rad_;
  state_msg.motor_a_velocity = motor_a_vel_rads_;
  state_msg.motor_b_position = motor_b_pos_rad_;
  state_msg.motor_b_velocity = motor_b_vel_rads_;
  state_msg.mode = mode_to_string(current_mode_);
  state_msg.homed = homed_;
  state_msg.homing_active = homing_active_;
  state_msg.homing_status = homing_status_;
  state_msg.move_done = !move_in_progress_;
  state_msg.enabled = motors_enabled_;
  state_msg.estop = estop_active_;
  state_msg.read_time_ms = last_read_ms_;
  state_msg.write_time_ms = last_write_ms_;
  state_msg.error_count = error_count_;
  state_msg.loop_rate_hz = measured_state_rate_hz_;
  state_pub_->publish(state_msg);

  // ── Joint states (for visualization / compatibility) ──
  auto js_msg = sensor_msgs::msg::JointState();
  js_msg.header.stamp = now();
  js_msg.name = {"motor_a_joint", "motor_b_joint"};
  js_msg.position = {motor_a_pos_rad_, motor_b_pos_rad_};
  js_msg.velocity = {motor_a_vel_rads_, motor_b_vel_rads_};
  joint_state_pub_->publish(js_msg);

  if (publish_odom_ || publish_odom_tf_) {
    publish_gantry_odometry(state_msg.header.stamp);
  }
}

void GantryController::reset_state_timer(double hz)
{
  hz = std::clamp(hz, 5.0, 120.0);
  stack_pose_publish_hz_ = hz;
  if (state_timer_) {
    state_timer_->cancel();
    state_timer_.reset();
  }
  const auto state_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(1.0 / hz));
  state_timer_ = create_wall_timer(
    state_period, std::bind(&GantryController::publish_state, this));
}

void GantryController::publish_state_if_due()
{
  // Never slower than 20 Hz so dashboard cart pose stays live.
  const double hz = std::clamp(stack_pose_publish_hz_, 20.0, 120.0);
  const auto now = std::chrono::steady_clock::now();
  const double since = std::chrono::duration<double>(
    now - last_state_publish_time_).count();
  if (since < (1.0 / hz)) {
    return;
  }
  publish_state();
}

void GantryController::pose_sync_hz_callback(
  const std_msgs::msg::Float64::SharedPtr msg)
{
  if (!stack_pose_sync_adaptive_ || !msg) {
    return;
  }
  const double hz = std::clamp(msg->data, 5.0, 120.0);
  if (std::abs(hz - stack_pose_publish_hz_) < 0.5) {
    return;
  }
  stack_pose_publish_hz_ = hz;
  RCLCPP_INFO(get_logger(),
              "Following /stack/pose_sync_hz → /gantry/state target %.1f Hz", hz);
}

void GantryController::publish_gantry_odometry(const rclcpp::Time & stamp)
{
  const double x = cart_x_m_;
  const double y = cart_y_m_;
  const double vx = cart_vx_ms_;
  const double vy = cart_vy_ms_;

  if (publish_odom_) {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_id_;
    odom.child_frame_id = odom_child_frame_id_;

    odom.pose.pose.position.x = x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation.w = 1.0;

    odom.twist.twist.linear.x = vx;
    odom.twist.twist.linear.y = vy;
    odom.twist.twist.linear.z = 0.0;

    // Planar XY gantry: trust x/y; fix z/yaw
    std::fill(odom.pose.covariance.begin(), odom.pose.covariance.end(), 0.0);
    odom.pose.covariance[0] = 1e-4;
    odom.pose.covariance[7] = 1e-4;
    odom.pose.covariance[14] = 1e6;
    odom.pose.covariance[21] = 1e6;
    odom.pose.covariance[28] = 1e6;
    odom.pose.covariance[35] = 1e6;

    std::fill(odom.twist.covariance.begin(), odom.twist.covariance.end(), 0.0);
    odom.twist.covariance[0] = 1e-4;
    odom.twist.covariance[7] = 1e-4;
    odom.twist.covariance[35] = 1e6;

    odom_pub_->publish(odom);
  }

  if (publish_odom_tf_ && tf_broadcaster_) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = odom_frame_id_;
    tf.child_frame_id = odom_child_frame_id_;
    tf.transform.translation.x = x;
    tf.transform.translation.y = y;
    tf.transform.translation.z = 0.0;
    tf.transform.rotation.w = 1.0;
    tf_broadcaster_->sendTransform(tf);
  }
}

// ============================================================================
// JOYSTICK CALLBACK
// ============================================================================
void GantryController::joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);

  // Button 10: toggle between JOG and ZV_JOG
  if (msg->buttons.size() > 9) {
    static bool last_btn10 = false;
    if (msg->buttons[9] && !last_btn10) {
      if (current_mode_ == Mode::JOG) {
        current_mode_ = Mode::ZV_JOG;
        zv_state_ = ZvState::IDLE;
        zv_button_held_ = false;
        RCLCPP_INFO(get_logger(), "Switched to ZV_JOG mode");
      } else if (current_mode_ == Mode::ZV_JOG) {
        current_mode_ = Mode::JOG;
        RCLCPP_INFO(get_logger(), "Switched to JOG mode");
      }
    }
    last_btn10 = msg->buttons[9];
  }

  // ZV_JOG mode: same controls as JOG (LB + left stick)
  if (current_mode_ == Mode::ZV_JOG) {
    if (msg->axes.size() < 2 || msg->buttons.size() < 6) return;

    // RB = estop
    if (msg->buttons[5]) {
      zv_state_ = ZvState::IDLE;
      zv_button_held_ = false;
      emergency_stop();
      return;
    }

    bool lb_held = msg->buttons[4];

    // Apply deadzone to stick
    auto deadzone = [](double v, double dz) -> double {
      if (std::abs(v) < dz) return 0.0;
      return (v - std::copysign(dz, v)) / (1.0 - dz);
    };
    double lx = deadzone(joy_axis_sign_x_ * msg->axes[0], 0.08);
    double ly = deadzone(joy_axis_sign_y_ * msg->axes[1], 0.08);
    bool stick_active = lb_held && (std::abs(lx) > 0.01 || std::abs(ly) > 0.01);

    if (stick_active && !zv_button_held_) {
      // Stick just activated — start ZV ramp up
      zv_dir_x_ = (std::abs(lx) > std::abs(ly)) ? (lx > 0 ? 1.0 : -1.0) : 0.0;
      zv_dir_y_ = (std::abs(ly) >= std::abs(lx)) ? (ly > 0 ? 1.0 : -1.0) : 0.0;
      zv_state_ = ZvState::RAMP_UP_1;
      zv_transition_time_ = std::chrono::steady_clock::now();
      RCLCPP_INFO(get_logger(), "ZV start: dir=(%.0f, %.0f)", zv_dir_x_, zv_dir_y_);
    }

    if (!stick_active && zv_button_held_) {
      // Stick just released — start ZV ramp down
      if (zv_state_ == ZvState::FULL_SPEED || zv_state_ == ZvState::RAMP_UP_1) {
        zv_state_ = ZvState::RAMP_DOWN_1;
        zv_transition_time_ = std::chrono::steady_clock::now();
        RCLCPP_INFO(get_logger(), "ZV stop: ramping down");
      }
    }

    zv_button_held_ = stick_active;
    last_cmd_time_ = std::chrono::steady_clock::now();
    return;
  }

  if (current_mode_ != Mode::JOG) return;
  if (msg->axes.size() < 2 || msg->buttons.size() < 6) { joy_enable_held_ = false; jog_vx_ = 0.0; jog_vy_ = 0.0; return; }

  // LB (button 4) = enable motion
  joy_enable_held_ = msg->buttons[4];

  // RB (button 5) = emergency stop
  if (msg->buttons[5]) {
    joy_enable_held_ = false;
    jog_vx_ = 0.0;
    jog_vy_ = 0.0;
    joy_enable_held_ = false;
    emergency_stop();
    return;
  }

  // D-pad or buttons to change speed preset
  // Y button (3) = cycle speed up, A button (0) = cycle speed down
  static bool last_y = false, last_a = false;
  if (msg->buttons[3] && !last_y) {
    jog_speed_preset_ = std::min(jog_speed_preset_ + 1, 2);
    RCLCPP_INFO(get_logger(), "Jog speed: %.0f mm/s", JOG_SPEEDS[jog_speed_preset_] * 1000);
  }
  if (msg->buttons[0] && !last_a) {
    jog_speed_preset_ = std::max(jog_speed_preset_ - 1, 0);
    RCLCPP_INFO(get_logger(), "Jog speed: %.0f mm/s", JOG_SPEEDS[jog_speed_preset_] * 1000);
  }
  last_y = msg->buttons[3];
  last_a = msg->buttons[0];

  if (joy_enable_held_) {
    double max_spd = JOG_SPEEDS[jog_speed_preset_];
    // Lab frame: origin bottom-left, +X right, +Y up (Logitech Dual Action)
    double lx = joy_axis_sign_x_ * msg->axes[0];
    double ly = joy_axis_sign_y_ * msg->axes[1];

    // Apply deadzone
    auto deadzone = [](double v, double dz) -> double {
      if (std::abs(v) < dz) return 0.0;
      return (v - std::copysign(dz, v)) / (1.0 - dz);
    };

    lx = deadzone(lx, 0.08);
    ly = deadzone(ly, 0.08);

    jog_vx_ = lx * max_spd;
    jog_vy_ = ly * max_spd;
    last_cmd_time_ = std::chrono::steady_clock::now();
  } else {
    jog_vx_ = 0.0;
    jog_vy_ = 0.0;
    joy_enable_held_ = false;
  }
}

// ============================================================================
// SERVICE CALLBACKS
// ============================================================================

void GantryController::set_mode_callback(
  const std::shared_ptr<gantry_control::srv::SetMode::Request> request,
  std::shared_ptr<gantry_control::srv::SetMode::Response> response)
{
  std::lock_guard<std::mutex> lock(state_mutex_);

  std::string mode_str = request->mode;

  // Stop current motion before switching
  if (motors_enabled_ && !estop_active_) {
    send_velocity_synced(0.0, 0.0);
    restore_motor_limits();
  }
  move_in_progress_ = false;

  if (mode_str == "IDLE") {
    current_mode_ = Mode::IDLE;
  } else if (mode_str == "HOME") {
    if (estop_active_) {
      response->success = false;
      response->message = "E-stop active. Clear E-stop first.";
      return;
    }
    if (homing_active_) {
      response->success = false;
      response->message = "Homing already in progress.";
      return;
    }
    if (!motors_enabled_) {
      if (!enable_motors()) {
        response->success = false;
        response->message = "Failed to enable motors for sensor homing.";
        return;
      }
    }
    begin_homing(PendingRunKind::ManualHome, Mode::IDLE);
    response->success = true;
    response->message = "Sensor homing started (seek back, then left).";
    RCLCPP_INFO(get_logger(), "Manual HOME — sensor homing");
    return;
  } else if (mode_str == "JOG") {
    if (!motors_enabled_) {
      response->success = false;
      response->message = "Motors not enabled.";
      return;
    }
    current_mode_ = Mode::JOG;
    jog_vx_ = 0.0;
    jog_vy_ = 0.0;
    joy_enable_held_ = false;
    jog_speed_preset_ = std::clamp<int>(request->jog_speed_preset, 0, 2);
    RCLCPP_INFO(get_logger(), "JOG mode, speed: %.0f mm/s",
                JOG_SPEEDS[jog_speed_preset_] * 1000);
  } else if (mode_str == "CSV") {
    if (!motors_enabled_) {
      response->success = false;
      response->message = "Motors not enabled.";
      return;
    }
    if (!load_csv_profile(request->csv_path)) {
      response->success = false;
      response->message = "Failed to load CSV: " + request->csv_path;
      return;
    }
    csv_index_ = 0;
    if (auto_home_before_run_ && !homed_) {
      begin_homing(PendingRunKind::Csv, Mode::CSV);
      RCLCPP_INFO(get_logger(),
                  "CSV loaded (%zu entries) — sensor homing before run",
                  csv_profile_.size());
    } else {
      current_mode_ = Mode::CSV;
      csv_start_time_ = std::chrono::steady_clock::now();
      RCLCPP_INFO(get_logger(), "CSV mode, loaded %zu entries from %s",
                  csv_profile_.size(), request->csv_path.c_str());
    }
  } else if (mode_str == "TRAJ") {
    clear_traj_profile();
    current_mode_ = Mode::TRAJ;
    traj_abort_ = false;
    move_in_progress_ = false;
    RCLCPP_INFO(get_logger(),
                "TRAJ mode — buffered WAYPOINTs or realtime STREAM on /traj_cmd; enable to start");
  } else if (mode_str == "MISSION") {
    if (!motors_enabled_) {
      response->success = false;
      response->message = "Motors not enabled.";
      return;
    }
    if (homing_active_) {
      response->success = false;
      response->message = "Homing in progress — wait before MISSION.";
      return;
    }
    if (auto_home_before_run_ && !homed_) {
      begin_homing(PendingRunKind::Mission, Mode::MISSION);
    } else {
      if (!homed_) {
        response->success = false;
        response->message = "Not homed. Call /gantry/set_mode with mode=HOME first.";
        return;
      }
      current_mode_ = Mode::MISSION;
    }
  } else if (mode_str == "ZV_JOG") {
    if (!motors_enabled_) {
      response->success = false;
      response->message = "Motors not enabled.";
      return;
    }
    current_mode_ = Mode::ZV_JOG;
    zv_state_ = ZvState::IDLE;
    zv_button_held_ = false;
    // Use jog_speed_preset for T and A override if provided
    RCLCPP_INFO(get_logger(), "ZV_JOG mode. T=%.3fs A=%.0fmm/s. D-pad left to move.", zv_T_, zv_A_*1000);
  } else {
    response->success = false;
    response->message = "Unknown mode: " + mode_str;
    return;
  }

  response->success = true;
  response->message = "Mode set to " + mode_to_string(current_mode_);
  RCLCPP_INFO(get_logger(), "Mode → %s", mode_to_string(current_mode_).c_str());
}

void GantryController::move_to_callback(
  const std::shared_ptr<gantry_control::srv::MoveTo::Request> request,
  std::shared_ptr<gantry_control::srv::MoveTo::Response> response)
{
  std::lock_guard<std::mutex> lock(state_mutex_);

  if (current_mode_ != Mode::MISSION) {
    response->success = false;
    response->message = "Not in MISSION mode.";
    return;
  }

  if (homing_active_) {
    response->success = false;
    response->message = "Homing in progress.";
    return;
  }

  if (!homed_) {
    response->success = false;
    response->message = "Not homed.";
    return;
  }

  double tx = request->x;
  double ty = request->y;

  const double lim = workspace_limit_enable_ ? workspace_limit_m_ : WORKSPACE_M;
  if (tx < 0.0 || ty < 0.0 || tx > lim || ty > lim) {
    response->success = false;
    char buf[160];
    snprintf(
      buf, sizeof(buf),
      "Target (%.3f, %.3f) outside workspace [0, %.2f] m (cart at %.3f, %.3f)",
      tx, ty, lim, cart_x_m_, cart_y_m_);
    response->message = buf;
    RCLCPP_WARN(get_logger(), "%s", buf);
    return;
  }

  mission_target_x_ = tx;
  mission_target_y_ = ty;

  constexpr double kAtTargetTolM = 0.008;
  if (std::hypot(cart_x_m_ - tx, cart_y_m_ - ty) <= kAtTargetTolM) {
    move_in_progress_ = false;
    response->success = true;
    char buf[128];
    snprintf(buf, sizeof(buf), "Already at (%.3f, %.3f) m", tx, ty);
    response->message = buf;
    RCLCPP_INFO(get_logger(), "%s", buf);
    return;
  }

  // Cart (m) → motor counts relative to home, then absolute encoder targets
  int32_t target_a, target_b;
  inverse_position(tx, ty, target_a, target_b);
  target_a += home_offset_a_;
  target_b += home_offset_b_;

  RCLCPP_INFO(
    get_logger(),
    "MoveTo cart=(%.3f, %.3f) m → motor counts A=%d B=%d (from cart 0,0)",
    tx, ty, target_a, target_b);

  {
    const double dx = tx - cart_x_m_;
    const double dy = ty - cart_y_m_;
    const double dist = std::hypot(dx, dy);
    double vx_lim = 0.0;
    double vy_lim = 0.0;
    if (dist > 1e-6) {
      vx_lim = mission_move_vel_ms_ * dx / dist;
      vy_lim = mission_move_vel_ms_ * dy / dist;
    } else {
      vx_lim = mission_move_vel_ms_;
    }
    apply_mission_move_limits(vx_lim, vy_lim);
  }

  send_position_synced(target_a, target_b);
  move_in_progress_ = true;

  response->success = true;
  char buf[128];
  snprintf(buf, sizeof(buf), "Moving to (%.3f, %.3f) m", tx, ty);
  response->message = buf;
  RCLCPP_INFO(get_logger(), "%s", buf);
}

void GantryController::estop_callback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  emergency_stop();
  response->success = true;
  response->message = "Emergency stop activated.";
}

void GantryController::clear_estop_callback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  clear_emergency_stop();
  payload_limit_tripped_ = false;
  workspace_limit_tripped_ = false;
  response->success = true;
  response->message = "Emergency stop cleared. Motors re-enabled.";
}

void GantryController::trip_payload_position_limit(
  const char * axis_label, double value_m)
{
  if (payload_limit_tripped_) {
    return;
  }
  payload_limit_tripped_ = true;
  RCLCPP_ERROR(
    get_logger(),
    "Payload vision limit exceeded: %s = %.1f mm (limit %.0f mm) — E-STOP",
    axis_label, value_m * 1000.0, payload_position_limit_m_ * 1000.0);
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    traj_abort_ = true;
    traj_running_ = false;
    clear_traj_profile();
    current_mode_ = Mode::IDLE;
  }
  emergency_stop();
}

void GantryController::payload_pose_callback(
  const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (!payload_limit_monitor_enable_ || payload_limit_tripped_ || estop_active_) {
    return;
  }
  if (!motors_enabled_) {
    return;
  }
  if (msg->data.size() < 5) {
    return;
  }

  const double limit = payload_position_limit_m_;
  struct AxisSample { const char * name; double value; };
  const AxisSample axes[] = {
    {"x1", msg->data[1]},
    {"z1", msg->data[2]},
    {"x2", msg->data[3]},
    {"z2", msg->data[4]},
  };

  for (const auto & axis : axes) {
    if (!std::isfinite(axis.value)) {
      continue;
    }
    if (std::abs(axis.value) > limit) {
      trip_payload_position_limit(axis.name, axis.value);
      return;
    }
  }
}

void GantryController::enable_callback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  if (enable_motors()) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (current_mode_ == Mode::TRAJ && !traj_running_ && !homing_active_) {
      if (traj_profile_loaded_ && !traj_profile_.empty()) {
        start_traj_execution();
        response->success = true;
        response->message = homing_active_
          ? "Motors enabled. Homing before TRAJ playback..."
          : "Motors enabled. TRAJ buffered playback started.";
      } else if (traj_realtime_enable_) {
        start_traj_realtime();
        response->success = true;
        response->message = homing_active_
          ? "Motors enabled. Homing before TRAJ realtime..."
          : "Motors enabled. TRAJ realtime (STREAM) active.";
      } else {
        response->success = true;
        response->message = "Motors enabled. Load TRAJ profile via /traj_cmd first.";
      }
    } else {
      response->success = true;
      response->message = "Motors enabled.";
    }
  } else {
    response->success = false;
    response->message = "Failed to enable motors.";
  }
}

void GantryController::disable_callback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  disable_motors();
  current_mode_ = Mode::IDLE;
  response->success = true;
  response->message = "Motors disabled.";
}

// ============================================================================
// MODE EXECUTORS
// ============================================================================

void GantryController::execute_idle()
{
  // Nothing to do — motors hold position via servo
}

bool GantryController::check_motor_homing_valid()
{
  if (!hardware_initialized_ || !node_a_ || !node_b_) {
    return false;
  }
  try {
    return node_a_->Motion.Homing.HomingValid() &&
           node_b_->Motion.Homing.HomingValid();
  } catch (...) {
    return false;
  }
}

void GantryController::begin_homing(PendingRunKind pending, Mode resume_mode)
{
  pending_run_kind_ = pending;
  mode_before_homing_ = resume_mode;
  homing_phase_ = HomingPhase::Validate;
  homing_active_ = true;
  homing_a_active_seen_ = false;
  homing_b_active_seen_ = false;
  reset_home_input_debounce();
  homing_status_ = "validate";
  homed_ = false;
  homing_cart_bias_x_ = 0.0;
  homing_cart_bias_y_ = 0.0;
  homing_deadline_ = std::chrono::steady_clock::now() +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(homing_timeout_s_));
  current_mode_ = Mode::HOMING;
  RCLCPP_INFO(get_logger(), "Sensor homing started (resume mode %s)",
              mode_to_string(resume_mode).c_str());
}

void GantryController::fail_homing(const std::string & reason)
{
  RCLCPP_ERROR(get_logger(), "Sensor homing failed: %s", reason.c_str());
  homing_status_ = "failed";
  homing_phase_ = HomingPhase::Failed;
  homing_active_ = false;
  homed_ = false;
  homing_cart_bias_x_ = 0.0;
  homing_cart_bias_y_ = 0.0;
  pending_run_kind_ = PendingRunKind::None;
  try {
    send_velocity_synced(0.0, 0.0);
  } catch (...) {}
  current_mode_ = Mode::IDLE;
  homing_phase_ = HomingPhase::Idle;
  error_count_++;
}

void GantryController::finalize_homing_success()
{
  double raw_x = 0.0;
  double raw_y = 0.0;
  try {
    home_offset_a_ = static_cast<int32_t>(node_a_->Motion.PosnMeasured.Value());
    home_offset_b_ = static_cast<int32_t>(node_b_->Motion.PosnMeasured.Value());

    const int32_t pos_a = static_cast<int32_t>(node_a_->Motion.PosnMeasured.Value()) - home_offset_a_;
    const int32_t pos_b = static_cast<int32_t>(node_b_->Motion.PosnMeasured.Value()) - home_offset_b_;
    motor_a_pos_rad_ = (pos_a / COUNTS_PER_REV) * 2.0 * M_PI;
    motor_b_pos_rad_ = (pos_b / COUNTS_PER_REV) * 2.0 * M_PI;

    forward_cartesian_unbiased(raw_x, raw_y);
  } catch (...) {
    fail_homing("encoder read after homing");
    return;
  }

  homed_ = true;
  homing_active_ = false;
  homing_status_ = "done";
  homing_phase_ = HomingPhase::Idle;
  current_mode_ = mode_before_homing_;

  // Software workspace origin at home sensors (physical FK rarely exactly 0,0).
  homing_cart_bias_x_ = raw_x;
  homing_cart_bias_y_ = raw_y;
  cart_x_m_ = 0.0;
  cart_y_m_ = 0.0;
  cart_vx_ms_ = 0.0;
  cart_vy_ms_ = 0.0;

  // Homing seek can trip workspace E-stop before bias is applied; safe to clear at home.
  workspace_limit_tripped_ = false;
  if (estop_active_) {
    clear_emergency_stop();
  }

  RCLCPP_INFO(get_logger(),
              "Sensor homing complete. Offsets A=%d B=%d software home (bias %.3f, %.3f) m → cart (0, 0)",
              home_offset_a_, home_offset_b_, homing_cart_bias_x_, homing_cart_bias_y_);

  dispatch_pending_run();
}

void GantryController::dispatch_pending_run()
{
  const auto kind = pending_run_kind_;
  pending_run_kind_ = PendingRunKind::None;

  switch (kind) {
    case PendingRunKind::TrajBuffered:
      do_start_traj_execution();
      break;
    case PendingRunKind::TrajRealtime:
      do_start_traj_realtime();
      break;
    case PendingRunKind::Csv:
      do_start_csv_run();
      break;
    case PendingRunKind::ManualHome:
    case PendingRunKind::Mission:
    case PendingRunKind::None:
    default:
      break;
  }
}

void GantryController::do_start_csv_run()
{
  csv_index_ = 0;
  csv_start_time_ = std::chrono::steady_clock::now();
  current_mode_ = Mode::CSV;
  RCLCPP_INFO(get_logger(), "CSV run started after homing");
}

void GantryController::reset_home_input_debounce()
{
  homing_left_debounce_count_ = 0;
  homing_back_debounce_count_ = 0;
}

bool GantryController::read_home_hub_input(
  size_t node_index, bool input_b, bool active_high)
{
  sFnd::INode * node = nullptr;
  if (node_index == 0) {
    node = node_a_;
  } else if (node_index == 1) {
    node = node_b_;
  } else {
    return false;
  }
  node->Status.RT.Refresh();
  const auto & cpm = node->Status.RT.Value().cpm;
  const bool raw = input_b ? (cpm.InB != 0) : (cpm.InA != 0);
  return active_high ? raw : !raw;
}

bool GantryController::home_left_sensor_active()
{
  return read_home_hub_input(
    home_left_node_, home_left_input_b_, home_left_input_active_high_);
}

bool GantryController::home_back_sensor_active()
{
  return read_home_hub_input(
    home_back_node_, home_back_input_b_, home_back_input_active_high_);
}

bool GantryController::home_sensors_both_active_debounced()
{
  if (home_left_sensor_active()) {
    homing_left_debounce_count_ =
      std::min(homing_left_debounce_count_ + 1, homing_input_debounce_ticks_ + 1);
  } else {
    homing_left_debounce_count_ = 0;
  }
  if (home_back_sensor_active()) {
    homing_back_debounce_count_ =
      std::min(homing_back_debounce_count_ + 1, homing_input_debounce_ticks_ + 1);
  } else {
    homing_back_debounce_count_ = 0;
  }
  return homing_left_debounce_count_ >= homing_input_debounce_ticks_ &&
         homing_back_debounce_count_ >= homing_input_debounce_ticks_;
}

void GantryController::execute_homing()
{
  if (!hardware_initialized_ || !node_a_ || !node_b_) {
    fail_homing("hardware not ready");
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  if (now > homing_deadline_) {
    fail_homing("timeout");
    return;
  }

  if (homing_mode_ == HomingMode::SensorInputs) {
    execute_homing_sensor_inputs();
    return;
  }
  execute_homing_teknic();
}

void GantryController::execute_homing_sensor_inputs()
{
  const double seek_speed = std::clamp(0.5 * mission_move_vel_ms_, 0.005, MAX_VEL_MS);
  const double vx_seek = -seek_speed;  // left
  const double vy_seek = -seek_speed;  // back

  try {
    switch (homing_phase_) {
      case HomingPhase::Validate:
        homing_status_ = "validate";
        homing_phase_ = HomingPhase::SensorCheck;
        break;

      case HomingPhase::SensorCheck:
        homing_status_ = "sensor_check";
        if (home_sensors_both_active_debounced()) {
          send_cartesian_velocity(0.0, 0.0);
          reset_home_input_debounce();
          RCLCPP_INFO(
            get_logger(),
            "Homing: both home sensors already active; releasing +X/+Y before re-approach");
          homing_phase_ = HomingPhase::ReleaseHomeSwitches;
          break;
        }
        reset_home_input_debounce();
        // Always seek back first (do not skip to left if back input reads idle/noisy).
        RCLCPP_INFO(get_logger(),
                    "Homing: seek back first (node %zu input %c), then left (node %zu input %c)",
                    home_back_node_, home_back_input_b_ ? 'B' : 'A',
                    home_left_node_, home_left_input_b_ ? 'B' : 'A');
        homing_phase_ = HomingPhase::SeekBack;
        break;

      case HomingPhase::ReleaseHomeSwitches:
        homing_status_ = "release_home";
        if (!(home_left_sensor_active() && home_back_sensor_active())) {
          send_cartesian_velocity(0.0, 0.0);
          reset_home_input_debounce();
          RCLCPP_INFO(
            get_logger(),
            "Homing: home switch released; seeking back then left");
          homing_phase_ = HomingPhase::SeekBack;
          break;
        }
        // Move gently into the workspace so already-active switches deassert,
        // then run the normal back→left approach to establish a fresh home edge.
        send_cartesian_velocity(seek_speed, seek_speed);
        break;

      case HomingPhase::SeekBack:
        homing_status_ = "seek_back";
        if (home_sensors_both_active_debounced()) {
          send_cartesian_velocity(0.0, 0.0);
          homing_phase_ = HomingPhase::Finalize;
          break;
        }
        {
          if (home_back_sensor_active()) {
            homing_back_debounce_count_ =
              std::min(homing_back_debounce_count_ + 1, homing_input_debounce_ticks_ + 1);
          } else {
            homing_back_debounce_count_ = 0;
          }
          if (homing_back_debounce_count_ >= homing_input_debounce_ticks_) {
            send_cartesian_velocity(0.0, 0.0);
            RCLCPP_INFO(get_logger(), "Back sensor active.");
            homing_left_debounce_count_ = 0;
            if (home_left_sensor_active()) {
              homing_phase_ = HomingPhase::Finalize;
            } else {
              homing_phase_ = HomingPhase::SeekLeft;
            }
            break;
          }
        }
        // Lab frame: origin bottom-left, +Y up → back = negative Y only (no X during back seek).
        send_cartesian_velocity(0.0, vy_seek);
        break;

      case HomingPhase::SeekLeft:
        homing_status_ = "seek_left";
        if (home_sensors_both_active_debounced()) {
          send_cartesian_velocity(0.0, 0.0);
          homing_phase_ = HomingPhase::Finalize;
          break;
        }
        {
          if (home_left_sensor_active()) {
            homing_left_debounce_count_ =
              std::min(homing_left_debounce_count_ + 1, homing_input_debounce_ticks_ + 1);
          } else {
            homing_left_debounce_count_ = 0;
          }
          if (homing_left_debounce_count_ >= homing_input_debounce_ticks_) {
            send_cartesian_velocity(0.0, 0.0);
            reset_home_input_debounce();
            if (home_sensors_both_active_debounced()) {
              RCLCPP_INFO(get_logger(), "Left sensor active — both home inputs on.");
              homing_phase_ = HomingPhase::Finalize;
            } else {
              fail_homing("left sensor reached but back sensor not active");
            }
            break;
          }
        }
        // Left = negative X only (after back sensor is satisfied).
        send_cartesian_velocity(vx_seek, 0.0);
        break;

      case HomingPhase::Finalize:
        send_cartesian_velocity(0.0, 0.0);
        if (!home_sensors_both_active_debounced()) {
          homing_status_ = "finalize_wait";
          break;
        }
        finalize_homing_success();
        break;

      case HomingPhase::Failed:
      case HomingPhase::Idle:
      default:
        send_cartesian_velocity(0.0, 0.0);
        break;
    }
  } catch (const std::exception & e) {
    fail_homing(e.what());
  } catch (...) {
    fail_homing("unknown exception during sensor homing");
  }
}

void GantryController::execute_homing_teknic()
{
  try {
    switch (homing_phase_) {
      case HomingPhase::Validate:
        homing_status_ = "validate";
        if (!check_motor_homing_valid()) {
          fail_homing(
            "HomingValid false — configure HOME_TO_SWITCH homing in Teknic ClearView for both motors");
          return;
        }
        homing_phase_ = HomingPhase::StartA;
        break;

      case HomingPhase::StartA:
        homing_status_ = "motor_a";
        RCLCPP_INFO(get_logger(), "Homing motor A (node 0)...");
        node_a_->Motion.Homing.SignalInvalid();
        node_a_->Motion.Homing.Initiate();
        homing_a_active_seen_ = false;
        homing_phase_ = HomingPhase::WaitA;
        break;

      case HomingPhase::WaitA: {
        homing_status_ = "motor_a";
        if (node_a_->Motion.Homing.IsHoming()) {
          homing_a_active_seen_ = true;
          return;
        }
        if (!homing_a_active_seen_) {
          return;
        }
        if (!node_a_->Motion.Homing.WasHomed()) {
          fail_homing("motor A did not home (check switch / ClearView direction)");
          return;
        }
        RCLCPP_INFO(get_logger(), "Motor A homed.");
        homing_phase_ = homing_sequential_ ? HomingPhase::StartB : HomingPhase::Finalize;
        break;
      }

      case HomingPhase::StartB:
        homing_status_ = "motor_b";
        RCLCPP_INFO(get_logger(), "Homing motor B (node 1)...");
        node_b_->Motion.Homing.SignalInvalid();
        node_b_->Motion.Homing.Initiate();
        homing_b_active_seen_ = false;
        homing_phase_ = HomingPhase::WaitB;
        break;

      case HomingPhase::WaitB: {
        homing_status_ = "motor_b";
        if (node_b_->Motion.Homing.IsHoming()) {
          homing_b_active_seen_ = true;
          return;
        }
        if (!homing_b_active_seen_) {
          return;
        }
        if (!node_b_->Motion.Homing.WasHomed()) {
          fail_homing("motor B did not home (check switch / ClearView direction)");
          return;
        }
        RCLCPP_INFO(get_logger(), "Motor B homed.");
        homing_phase_ = HomingPhase::Finalize;
        break;
      }

      case HomingPhase::Finalize:
        finalize_homing_success();
        break;

      case HomingPhase::Failed:
      case HomingPhase::Idle:
      case HomingPhase::SensorCheck:
      case HomingPhase::SeekBack:
      case HomingPhase::SeekLeft:
      default:
        break;
    }
  } catch (const std::exception & e) {
    fail_homing(e.what());
  } catch (...) {
    fail_homing("unknown exception during teknic homing");
  }
}

void GantryController::execute_jog()
{
  // Watchdog: if no joystick input for timeout, stop
  auto elapsed = std::chrono::steady_clock::now() - last_cmd_time_;
  double elapsed_s = std::chrono::duration<double>(elapsed).count();

  double vx = jog_vx_;
  double vy = jog_vy_;

  if (elapsed_s > WATCHDOG_TIMEOUT_S) {
    vx = 0.0;
    vy = 0.0;
  }

  // CoreXY inverse kinematics → motor RPM
  send_cartesian_velocity(vx, vy);
}

void GantryController::execute_zv_jog()
{
  // Update parameters dynamically
  zv_T_ = get_parameter("zv_T").as_double();

  double vx = 0.0, vy = 0.0;

  switch (zv_state_) {
    case ZvState::IDLE:
      vx = 0.0;
      vy = 0.0;
      break;

    case ZvState::RAMP_UP_1: {
      // Half speed for T seconds
      vx = zv_dir_x_ * 0.5 * JOG_SPEEDS[jog_speed_preset_];
      vy = zv_dir_y_ * 0.5 * JOG_SPEEDS[jog_speed_preset_];
      auto elapsed = std::chrono::steady_clock::now() - zv_transition_time_;
      if (std::chrono::duration<double>(elapsed).count() >= zv_T_) {
        zv_state_ = ZvState::FULL_SPEED;
        RCLCPP_INFO(get_logger(), "ZV: full speed");
      }
      break;
    }

    case ZvState::FULL_SPEED:
      // Full speed until button released
      vx = zv_dir_x_ * JOG_SPEEDS[jog_speed_preset_];
      vy = zv_dir_y_ * JOG_SPEEDS[jog_speed_preset_];
      break;

    case ZvState::RAMP_DOWN_1: {
      // Half speed for T seconds
      vx = zv_dir_x_ * 0.5 * JOG_SPEEDS[jog_speed_preset_];
      vy = zv_dir_y_ * 0.5 * JOG_SPEEDS[jog_speed_preset_];
      auto elapsed = std::chrono::steady_clock::now() - zv_transition_time_;
      if (std::chrono::duration<double>(elapsed).count() >= zv_T_) {
        zv_state_ = ZvState::IDLE;
        vx = 0.0;
        vy = 0.0;
        RCLCPP_INFO(get_logger(), "ZV: stopped");
      }
      break;
    }
  }

  send_cartesian_velocity(vx, vy);
}

void GantryController::execute_csv()
{
  if (homing_active_) {
    return;
  }

  auto elapsed = std::chrono::steady_clock::now() - csv_start_time_;
  double t = std::chrono::duration<double>(elapsed).count();

  // Find current CSV entry
  while (csv_index_ < csv_profile_.size() - 1 &&
         csv_profile_[csv_index_ + 1].time_s <= t) {
    csv_index_++;
  }

  if (csv_index_ >= csv_profile_.size()) {
    // Profile complete — stop
    send_velocity_synced(0.0, 0.0);
    current_mode_ = Mode::IDLE;
    RCLCPP_INFO(get_logger(), "CSV profile complete.");
    return;
  }

  // Step input: use the velocity from the current entry directly (no interpolation)
  double vx_ms = csv_profile_[csv_index_].vx_mm_s / 1000.0;
  double vy_ms = csv_profile_[csv_index_].vy_mm_s / 1000.0;

  RCLCPP_INFO(get_logger(), "CSV t=%.3f vy=%.1f", t, vy_ms * 1000);
  send_cartesian_velocity(vx_ms, vy_ms);
}

void GantryController::clear_traj_profile()
{
  traj_profile_.clear();
  traj_index_ = 0;
  traj_profile_loaded_ = false;
  traj_running_ = false;
  traj_realtime_active_ = false;
  stream_vx_mm_s_ = 0.0;
  stream_vy_mm_s_ = 0.0;
}

void GantryController::publish_motion_start()
{
  gantry_control::msg::TrajCmd msg;
  msg.header.stamp = now();
  msg.command = gantry_control::msg::TrajCmd::MOTION_START;
  msg.time_s = 0.0;
  traj_cmd_pub_->publish(msg);
}

void GantryController::publish_traj_playback(double profile_time_s)
{
  if (traj_profile_.empty() || traj_index_ >= traj_profile_.size()) {
    return;
  }

  const auto & seg = traj_profile_[traj_index_];
  gantry_control::msg::TrajCmd msg;
  msg.header.stamp = now();
  msg.command = gantry_control::msg::TrajCmd::PLAYBACK;
  msg.time_s = profile_time_s;
  msg.x = cart_x_m_;
  msg.y = cart_y_m_;
  msg.vx_mm_s = seg.vx_mm_s;
  msg.vy_mm_s = seg.vy_mm_s;
  msg.motor_a_pos_rad = motor_a_pos_rad_;
  msg.motor_b_pos_rad = motor_b_pos_rad_;
  msg.motor_a_vel_rad_s = motor_a_vel_rads_;
  msg.motor_b_vel_rad_s = motor_b_vel_rads_;
  traj_cmd_pub_->publish(msg);
}

void GantryController::do_start_traj_execution()
{
  traj_realtime_active_ = false;
  traj_index_ = 0;
  traj_start_time_ = std::chrono::steady_clock::now();
  traj_running_ = true;
  publish_motion_start();
  RCLCPP_INFO(get_logger(),
              "TRAJ buffered playback started (%zu segments) — profile time t=0",
              traj_profile_.size());
}

void GantryController::do_start_traj_realtime()
{
  traj_realtime_active_ = true;
  traj_index_ = 0;
  traj_start_time_ = std::chrono::steady_clock::now();
  traj_running_ = true;
  stream_vx_mm_s_ = 0.0;
  stream_vy_mm_s_ = 0.0;
  last_cmd_time_ = std::chrono::steady_clock::now();
  publish_motion_start();
  RCLCPP_INFO(get_logger(),
              "TRAJ realtime started — publish STREAM on /traj_cmd (vx_mm_s, vy_mm_s)");
}

void GantryController::start_traj_execution()
{
  if (traj_profile_.empty()) {
    RCLCPP_WARN(get_logger(), "TRAJ profile empty — cannot start playback");
    return;
  }
  if (homing_active_) {
    RCLCPP_WARN(get_logger(), "Homing in progress — TRAJ start deferred");
    return;
  }
  if (auto_home_before_run_ && !homed_) {
    begin_homing(PendingRunKind::TrajBuffered, Mode::TRAJ);
    return;
  }
  do_start_traj_execution();
}

void GantryController::start_traj_realtime()
{
  if (homing_active_) {
    RCLCPP_WARN(get_logger(), "Homing in progress — TRAJ realtime deferred");
    return;
  }
  if (auto_home_before_run_ && !homed_) {
    begin_homing(PendingRunKind::TrajRealtime, Mode::TRAJ);
    return;
  }
  do_start_traj_realtime();
}

void GantryController::publish_traj_playback_realtime(
  double profile_time_s, double vx_mm_s, double vy_mm_s)
{
  gantry_control::msg::TrajCmd msg;
  msg.header.stamp = now();
  msg.command = gantry_control::msg::TrajCmd::PLAYBACK;
  msg.time_s = profile_time_s;
  msg.x = cart_x_m_;
  msg.y = cart_y_m_;
  msg.vx_mm_s = vx_mm_s;
  msg.vy_mm_s = vy_mm_s;
  msg.motor_a_pos_rad = motor_a_pos_rad_;
  msg.motor_b_pos_rad = motor_b_pos_rad_;
  msg.motor_a_vel_rad_s = motor_a_vel_rads_;
  msg.motor_b_vel_rad_s = motor_b_vel_rads_;
  traj_cmd_pub_->publish(msg);
}

void GantryController::traj_cmd_callback(
  const gantry_control::msg::TrajCmd::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);

  if (current_mode_ != Mode::TRAJ) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Ignoring /traj_cmd — not in TRAJ mode");
    return;
  }

  last_cmd_time_ = std::chrono::steady_clock::now();

  if (msg->command == gantry_control::msg::TrajCmd::ABORT) {
    traj_abort_ = true;
    traj_running_ = false;
    clear_traj_profile();
    send_velocity_synced(0.0, 0.0);
    current_mode_ = Mode::IDLE;
    RCLCPP_WARN(get_logger(), "TRAJ abort → IDLE");
    return;
  }

  if (msg->command == gantry_control::msg::TrajCmd::MOTION_START ||
      msg->command == gantry_control::msg::TrajCmd::PLAYBACK) {
    return;
  }

  if (msg->command == gantry_control::msg::TrajCmd::PROFILE_START) {
    clear_traj_profile();
    traj_realtime_active_ = false;
    RCLCPP_INFO(get_logger(), "TRAJ profile load started");
    return;
  }

  if (msg->command == gantry_control::msg::TrajCmd::PROFILE_DONE) {
    traj_profile_loaded_ = true;
    traj_realtime_active_ = false;
    RCLCPP_INFO(get_logger(), "TRAJ profile loaded (%zu segments)",
                traj_profile_.size());
    RCLCPP_INFO(get_logger(),
                "Profile times are relative to motion start — call /gantry/enable");
    return;
  }

  if (msg->command == gantry_control::msg::TrajCmd::STREAM) {
    stream_vx_mm_s_ = msg->vx_mm_s;
    stream_vy_mm_s_ = msg->vy_mm_s;
    last_cmd_time_ = std::chrono::steady_clock::now();
    if (!traj_running_ && motors_enabled_ && traj_realtime_enable_) {
      start_traj_realtime();
    }
    return;
  }

  if (msg->command == gantry_control::msg::TrajCmd::WAYPOINT) {
    CsvEntry entry;
    entry.time_s = msg->time_s;
    entry.vx_mm_s = msg->vx_mm_s;
    entry.vy_mm_s = msg->vy_mm_s;
    traj_profile_.push_back(entry);
    RCLCPP_INFO(get_logger(),
                "TRAJ segment[%zu] t=%.3fs v=(%.1f, %.1f) mm/s pos=(%.3f, %.3f) m",
                traj_profile_.size(), entry.time_s,
                entry.vx_mm_s, entry.vy_mm_s, msg->x, msg->y);
    return;
  }

  RCLCPP_WARN(get_logger(), "Unknown /traj_cmd command: %u", msg->command);
}

void GantryController::execute_traj()
{
  if (traj_abort_) {
    send_velocity_synced(0.0, 0.0);
    traj_running_ = false;
    traj_abort_ = false;
    clear_traj_profile();
    current_mode_ = Mode::IDLE;
    RCLCPP_WARN(get_logger(), "TRAJ aborted → IDLE");
    return;
  }

  if (!motors_enabled_) {
    return;
  }

  if (!traj_running_) {
    return;
  }

  auto elapsed = std::chrono::steady_clock::now() - traj_start_time_;
  double t = std::chrono::duration<double>(elapsed).count();

  // ── Real-time STREAM (research nodes) ──
  if (traj_realtime_active_) {
    double cmd_age = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - last_cmd_time_).count();
    double vx_ms = 0.0;
    double vy_ms = 0.0;
    if (cmd_age <= traj_stream_timeout_s_) {
      vx_ms = stream_vx_mm_s_ / 1000.0;
      vy_ms = stream_vy_mm_s_ / 1000.0;
    }
    publish_traj_playback_realtime(t, stream_vx_mm_s_, stream_vy_mm_s_);
    send_cartesian_velocity(vx_ms, vy_ms);
    return;
  }

  // ── Buffered profile playback ──
  if (!traj_profile_loaded_ || traj_profile_.empty()) {
    return;
  }

  if (t > traj_profile_.back().time_s) {
    send_velocity_synced(0.0, 0.0);
    traj_running_ = false;
    traj_profile_loaded_ = false;
    clear_traj_profile();
    current_mode_ = Mode::IDLE;
    RCLCPP_INFO(get_logger(), "TRAJ profile complete → IDLE");
    return;
  }

  while (traj_index_ < traj_profile_.size() - 1 &&
         traj_profile_[traj_index_ + 1].time_s <= t) {
    traj_index_++;
  }

  publish_traj_playback(t);

  double vx_ms = traj_profile_[traj_index_].vx_mm_s / 1000.0;
  double vy_ms = traj_profile_[traj_index_].vy_mm_s / 1000.0;
  send_cartesian_velocity(vx_ms, vy_ms);
}

void GantryController::execute_mission()
{
  if (!move_in_progress_) return;

  // Check if position move is done
  if (is_move_done()) {
    move_in_progress_ = false;
    restore_motor_limits();
    RCLCPP_INFO(get_logger(), "Move done. Position: (%.3f, %.3f) m", cart_x_m_, cart_y_m_);
  }
}

// ============================================================================
// HARDWARE INTERFACE
// ============================================================================

bool GantryController::init_hardware()
{
  try {
    sys_mgr_ = sFnd::SysManager::Instance();

    std::vector<std::string> comHubPorts;
    sFnd::SysManager::FindComHubPorts(comHubPorts);

    if (comHubPorts.empty()) {
      RCLCPP_ERROR(get_logger(), "No SC4-HUB found");
      return false;
    }

    RCLCPP_INFO(get_logger(), "Found SC4-HUB on %s", comHubPorts[0].c_str());
    sys_mgr_->ComHubPort(0, comHubPorts[0].c_str());
    sys_mgr_->PortsOpen(1);

    port_ = &sys_mgr_->Ports(0);
    std::this_thread::sleep_for(100ms);

    if (port_->OpenState() != OPENED_ONLINE) {
      RCLCPP_ERROR(get_logger(), "Port failed to open");
      return false;
    }

    if (port_->NodeCount() < 2) {
      RCLCPP_ERROR(get_logger(), "Expected 2 motors, found %d", port_->NodeCount());
      return false;
    }

    node_a_ = &port_->Nodes(0);
    node_b_ = &port_->Nodes(1);

    RCLCPP_INFO(get_logger(), "Motor A: S/N %u", node_a_->Info.SerialNumber.Value());
    RCLCPP_INFO(get_logger(), "Motor B: S/N %u", node_b_->Info.SerialNumber.Value());

    const bool homing_a = node_a_->Motion.Homing.HomingValid();
    const bool homing_b = node_b_->Motion.Homing.HomingValid();
    RCLCPP_INFO(get_logger(), "HomingValid: motor A=%s motor B=%s",
                homing_a ? "yes" : "no", homing_b ? "yes" : "no");
    if (homing_mode_ == HomingMode::TeknicHoming && (!homing_a || !homing_b)) {
      RCLCPP_WARN(get_logger(),
                  "Teknic homing not configured in ClearView — set HOME_TO_SWITCH or use homing_mode:=sensor_inputs");
    }
    if (homing_mode_ == HomingMode::SensorInputs) {
      RCLCPP_INFO(get_logger(),
                  "Sensor-input homing: left=SC4 %zu%c back=%zu%c (no ClearView homing save)",
                  home_left_node_, home_left_input_b_ ? 'B' : 'A',
                  home_back_node_, home_back_input_b_ ? 'B' : 'A');
    }

    // Assign both motors to the same trigger group for synchronized moves
    node_a_->Motion.Adv.TriggerGroup(SYNC_TRIGGER_GROUP);
    node_b_->Motion.Adv.TriggerGroup(SYNC_TRIGGER_GROUP);
    RCLCPP_INFO(get_logger(), "Motors assigned to trigger group %zu", SYNC_TRIGGER_GROUP);

    hardware_initialized_ = true;
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Hardware init error: %s", e.what());
    return false;
  } catch (...) {
    RCLCPP_ERROR(get_logger(), "Hardware init failed (unknown error)");
    return false;
  }
}

void GantryController::shutdown_hardware()
{
  if (sys_mgr_ && port_ && port_->OpenState() == OPENED_ONLINE) {
    sys_mgr_->PortsClose();
  }
  hardware_initialized_ = false;
}

bool GantryController::enable_motors()
{
  if (!hardware_initialized_) return false;
  try {
    // Clear alerts and stops
    node_a_->Status.AlertsClear();
    node_b_->Status.AlertsClear();
    node_a_->Motion.NodeStopClear();
    node_b_->Motion.NodeStopClear();

    // Enable
    node_a_->EnableReq(true);
    node_b_->EnableReq(true);
    std::this_thread::sleep_for(200ms);

    // Configure units and limits
    node_a_->VelUnit(sFnd::INode::RPM);
    node_a_->AccUnit(sFnd::INode::RPM_PER_SEC);
    node_a_->Motion.AccLimit = MAX_ACCEL_RPM_S;
    node_a_->Motion.VelLimit = MAX_MOTOR_RPM;

    node_b_->VelUnit(sFnd::INode::RPM);
    node_b_->AccUnit(sFnd::INode::RPM_PER_SEC);
    node_b_->Motion.AccLimit = MAX_ACCEL_RPM_S;
    node_b_->Motion.VelLimit = MAX_MOTOR_RPM;

    motors_enabled_ = true;
    RCLCPP_INFO(get_logger(), "Motors enabled. AccLimit=%.0f RPM/s, VelLimit=%.0f RPM",
                MAX_ACCEL_RPM_S, MAX_MOTOR_RPM);
    return true;
  } catch (...) {
    RCLCPP_ERROR(get_logger(), "Failed to enable motors");
    error_count_++;
    return false;
  }
}

void GantryController::disable_motors()
{
  try {
    if (node_a_) node_a_->EnableReq(false);
    if (node_b_) node_b_->EnableReq(false);
  } catch (...) {}
  motors_enabled_ = false;
  RCLCPP_INFO(get_logger(), "Motors disabled.");
}

void GantryController::emergency_stop()
{
  estop_active_ = true;
  try {
    if (node_a_) node_a_->Motion.NodeStop(STOP_TYPE_ABRUPT);
    if (node_b_) node_b_->Motion.NodeStop(STOP_TYPE_ABRUPT);
  } catch (...) {}
  current_mode_ = Mode::IDLE;
  RCLCPP_WARN(get_logger(), "EMERGENCY STOP");
}

void GantryController::clear_emergency_stop()
{
  if (!hardware_initialized_) return;
  try {
    node_a_->Status.AlertsClear();
    node_b_->Status.AlertsClear();
    node_a_->Motion.NodeStopClear();
    node_b_->Motion.NodeStopClear();
    node_a_->EnableReq(true);
    node_b_->EnableReq(true);
    std::this_thread::sleep_for(100ms);
  } catch (...) {
    error_count_++;
  }
  estop_active_ = false;
  motors_enabled_ = true;
  RCLCPP_INFO(get_logger(), "E-stop cleared, motors re-enabled.");
}

// ============================================================================
// ENCODER READING
// ============================================================================
void GantryController::forward_cartesian_unbiased(double & x_m, double & y_m)
{
  forward_kinematics(motor_a_pos_rad_, motor_b_pos_rad_, x_m, y_m);
}

void GantryController::read_encoders()
{
  try {
    // Read raw encoder counts
    int32_t raw_a = static_cast<int32_t>(node_a_->Motion.PosnMeasured.Value());
    int32_t raw_b = static_cast<int32_t>(node_b_->Motion.PosnMeasured.Value());

    // Apply home offsets
    int32_t pos_a = raw_a - home_offset_a_;
    int32_t pos_b = raw_b - home_offset_b_;

    // Convert to radians
    motor_a_pos_rad_ = (pos_a / COUNTS_PER_REV) * 2.0 * M_PI;
    motor_b_pos_rad_ = (pos_b / COUNTS_PER_REV) * 2.0 * M_PI;

    // Read velocities (RPM → rad/s)
    double vel_a_rpm = node_a_->Motion.VelMeasured.Value();
    double vel_b_rpm = node_b_->Motion.VelMeasured.Value();
    motor_a_vel_rads_ = vel_a_rpm * 2.0 * M_PI / 60.0;
    motor_b_vel_rads_ = vel_b_rpm * 2.0 * M_PI / 60.0;

    // Compute Cartesian state (workspace frame; home sensors → 0,0 after bias)
    forward_kinematics(motor_a_pos_rad_, motor_b_pos_rad_, cart_x_m_, cart_y_m_);
    forward_velocity(motor_a_vel_rads_, motor_b_vel_rads_, cart_vx_ms_, cart_vy_ms_);
    cart_x_m_ -= homing_cart_bias_x_;
    cart_y_m_ -= homing_cart_bias_y_;

    if (homed_ && std::abs(cart_x_m_) < 0.015 && std::abs(cart_y_m_) < 0.015) {
      cart_x_m_ = 0.0;
      cart_y_m_ = 0.0;
    }

  } catch (...) {
    error_count_++;
  }
}

// ============================================================================
// SYNCHRONIZED MOTOR COMMANDS
// ============================================================================

void GantryController::trip_workspace_limit(const char * axis_label, double value_m)
{
  if (workspace_limit_tripped_) {
    return;
  }
  workspace_limit_tripped_ = true;
  RCLCPP_ERROR(
    get_logger(),
    "Workspace limit exceeded: %s = %.1f mm (limit ±%.0f mm) — E-STOP",
    axis_label, value_m * 1000.0, workspace_limit_m_ * 1000.0);
  emergency_stop();
  traj_running_ = false;
  clear_traj_profile();
  current_mode_ = Mode::IDLE;
}

bool GantryController::check_workspace_position()
{
  if (homing_active_ || current_mode_ == Mode::HOMING) {
    return true;
  }
  if (!workspace_limit_enable_ || workspace_limit_tripped_ || estop_active_) {
    return true;
  }
  const double lim = workspace_limit_m_;
  if (std::abs(cart_x_m_) > lim) {
    trip_workspace_limit("x", cart_x_m_);
    return false;
  }
  if (std::abs(cart_y_m_) > lim) {
    trip_workspace_limit("y", cart_y_m_);
    return false;
  }
  return true;
}

void GantryController::clamp_velocity_to_workspace(double & vx_ms, double & vy_ms)
{
  if (!workspace_limit_enable_) {
    return;
  }
  const double lim = workspace_limit_m_;
  const double margin = 0.002;  // 2 mm guard band

  if (cart_x_m_ >= lim - margin && vx_ms > 0.0) {
    vx_ms = 0.0;
  }
  if (cart_x_m_ <= -lim + margin && vx_ms < 0.0) {
    vx_ms = 0.0;
  }
  if (cart_y_m_ >= lim - margin && vy_ms > 0.0) {
    vy_ms = 0.0;
  }
  if (cart_y_m_ <= -lim + margin && vy_ms < 0.0) {
    vy_ms = 0.0;
  }
}

bool GantryController::send_cartesian_velocity(double vx_ms, double vy_ms)
{
  if (!check_workspace_position()) {
    return false;
  }
  clamp_velocity_to_workspace(vx_ms, vy_ms);
  vx_ms = std::clamp(vx_ms, -MAX_VEL_MS, MAX_VEL_MS);
  vy_ms = std::clamp(vy_ms, -MAX_VEL_MS, MAX_VEL_MS);
  double rpm_a, rpm_b;
  inverse_velocity(vx_ms, vy_ms, rpm_a, rpm_b);
  send_velocity_synced(rpm_a, rpm_b);
  return true;
}

void GantryController::send_velocity_synced(double vel_a_rpm, double vel_b_rpm)
{
  if (!hardware_initialized_ || !motors_enabled_ || estop_active_) return;

  try {
    // Clamp
    vel_a_rpm = std::clamp(vel_a_rpm, -MAX_MOTOR_RPM, MAX_MOTOR_RPM);
    vel_b_rpm = std::clamp(vel_b_rpm, -MAX_MOTOR_RPM, MAX_MOTOR_RPM);

    // Low-level immediate velocity — raw, no motor ramping
    double vel_a_cps = vel_a_rpm * COUNTS_PER_REV / 60.0;
    double vel_b_cps = vel_b_rpm * COUNTS_PER_REV / 60.0;
    mgVelStyle style;
    style.styleCode = MG_MOVE_VEL_IMMEDIATE;
    cpmForkMoveVelEx(node_a_->Info.Ex.Addr(), vel_a_cps, 0, style);
    cpmForkMoveVelEx(node_b_->Info.Ex.Addr(), vel_b_cps, 0, style);

  } catch (...) {
    error_count_++;
  }
}

void GantryController::apply_mission_move_limits(double vx_ms, double vy_ms)
{
  if (!hardware_initialized_ || !motors_enabled_) {
    return;
  }

  double rpm_a = 0.0;
  double rpm_b = 0.0;
  inverse_velocity(vx_ms, vy_ms, rpm_a, rpm_b);
  const double vel_lim =
    std::clamp(std::max(std::abs(rpm_a), std::abs(rpm_b)), 20.0, MAX_MOTOR_RPM);

  double acc_a = 0.0;
  double acc_b = 0.0;
  inverse_velocity(mission_move_accel_ms2_, mission_move_accel_ms2_, acc_a, acc_b);
  const double acc_lim =
    std::clamp(std::max(std::abs(acc_a), std::abs(acc_b)), 100.0, MAX_ACCEL_RPM_S);

  try {
    node_a_->Motion.VelLimit = vel_lim;
    node_b_->Motion.VelLimit = vel_lim;
    node_a_->Motion.AccLimit = acc_lim;
    node_b_->Motion.AccLimit = acc_lim;
  } catch (...) {
    error_count_++;
  }
}

void GantryController::restore_motor_limits()
{
  if (!hardware_initialized_ || !motors_enabled_) {
    return;
  }
  try {
    node_a_->Motion.VelLimit = MAX_MOTOR_RPM;
    node_b_->Motion.VelLimit = MAX_MOTOR_RPM;
    node_a_->Motion.AccLimit = MAX_ACCEL_RPM_S;
    node_b_->Motion.AccLimit = MAX_ACCEL_RPM_S;
  } catch (...) {
    error_count_++;
  }
}

void GantryController::send_position_synced(int32_t pos_a_counts, int32_t pos_b_counts)
{
  if (!hardware_initialized_ || !motors_enabled_ || estop_active_) return;

  try {
    // Clear previous move done flags
    node_a_->Motion.MoveWentDone();
    node_b_->Motion.MoveWentDone();

    // Send triggered position moves (absolute, queued)
    node_a_->Motion.Adv.MovePosnStart(pos_a_counts, true, true, false);
    node_b_->Motion.Adv.MovePosnStart(pos_b_counts, true, true, false);

    // Fire both simultaneously
    port_->Adv.TriggerMovesInGroup(SYNC_TRIGGER_GROUP);

  } catch (...) {
    error_count_++;
  }
}

void GantryController::send_velocity_immediate(size_t motor_idx, double vel_cps)
{
  if (!hardware_initialized_ || !motors_enabled_ || estop_active_) return;
  try {
    sFnd::INode * node = (motor_idx == 0) ? node_a_ : node_b_;
    mgVelStyle style;
    style.styleCode = MG_MOVE_VEL_STYLE_TRIG;
    cpmForkMoveVelEx(node->Info.Ex.Addr(), vel_cps, 0, style);
  } catch (...) {
    error_count_++;
  }
}

bool GantryController::is_move_done()
{
  try {
    return node_a_->Motion.MoveIsDone() && node_b_->Motion.MoveIsDone();
  } catch (...) {
    error_count_++;
    return false;
  }
}

// ============================================================================
// COREXY KINEMATICS
// ============================================================================

void GantryController::forward_kinematics(
  double motor_a_rad, double motor_b_rad,
  double & x_m, double & y_m)
{
  // Motor radians → linear displacement at belt
  double la = motor_a_rad * R_EFF;  // linear A
  double lb = motor_b_rad * R_EFF;  // linear B

  // CoreXY forward: derived from inverse
  // inverse: la = -x - y, lb = -x + y
  // forward: x = -(la + lb)/2, y = (lb - la)/2
  x_m = -(la + lb) / 2.0;
  y_m = (lb - la) / 2.0;
}

void GantryController::forward_velocity(
  double motor_a_rads, double motor_b_rads,
  double & vx_ms, double & vy_ms)
{
  double va = motor_a_rads * R_EFF;
  double vb = motor_b_rads * R_EFF;
  vx_ms = -(va + vb) / 2.0;
  vy_ms = (vb - va) / 2.0;
}

void GantryController::inverse_velocity(
  double vx_ms, double vy_ms,
  double & motor_a_rpm, double & motor_b_rpm)
{
  // Cartesian m/s → motor linear velocity m/s
  // la = -vx - vy, lb = -vx + vy
  double la = -vx_ms - vy_ms;
  double lb = -vx_ms + vy_ms;

  // Linear → angular (rad/s) at motor shaft
  double wa = la / R_EFF;
  double wb = lb / R_EFF;

  // rad/s → RPM
  motor_a_rpm = wa * 60.0 / (2.0 * M_PI);
  motor_b_rpm = wb * 60.0 / (2.0 * M_PI);
}

void GantryController::inverse_position(
  double x_m, double y_m,
  int32_t & motor_a_counts, int32_t & motor_b_counts)
{
  // Cartesian meters → motor linear displacement
  double la = -x_m - y_m;
  double lb = -x_m + y_m;

  // Linear → motor radians
  double ra = la / R_EFF;
  double rb = lb / R_EFF;

  // Radians → encoder counts (absolute, relative to home)
  motor_a_counts = static_cast<int32_t>(ra * COUNTS_PER_REV / (2.0 * M_PI));
  motor_b_counts = static_cast<int32_t>(rb * COUNTS_PER_REV / (2.0 * M_PI));
}

// ============================================================================
// CSV PROFILE LOADER
// ============================================================================

bool GantryController::load_csv_profile(const std::string & path)
{
  csv_profile_.clear();

  std::ifstream file(path);
  if (!file.is_open()) {
    RCLCPP_ERROR(get_logger(), "Cannot open CSV: %s", path.c_str());
    return false;
  }

  std::string line;
  // Skip header line
  std::getline(file, line);

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream ss(line);
    CsvEntry entry;
    char comma;
    if (ss >> entry.time_s >> comma >> entry.vx_mm_s >> comma >> entry.vy_mm_s) {
      csv_profile_.push_back(entry);
    }
  }

  if (csv_profile_.empty()) {
    RCLCPP_ERROR(get_logger(), "CSV file empty or invalid format");
    return false;
  }

  RCLCPP_INFO(get_logger(), "CSV: %zu entries, %.3f to %.3f sec",
              csv_profile_.size(), csv_profile_.front().time_s, csv_profile_.back().time_s);
  return true;
}

}  // namespace gantry_control

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<gantry_control::GantryController>();
  rclcpp::on_shutdown([node]() {
    if (node) {
      node->shutdown_for_exit();
    }
  });
  try {
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node->get_logger(), "spin exception: %s", e.what());
  }
  node->shutdown_for_exit();
  rclcpp::shutdown();
  return 0;
}
