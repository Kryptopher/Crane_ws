"""
Mission planner stack: gantry + vision + encoders + crane ops dashboard.

Starts the crane stack and serves crane_ops.html on port 8081.
Camera tracking and the separate initial-condition cart are temporarily
opt-in to keep the default Mission Planner launch lightweight.

  ros2 launch gantry_control mission_planner.launch.py
  ros2 launch gantry_control mission_planner.launch.py start_phase1_tracker:=true oak_ip:=192.168.0.153
  ros2 launch gantry_control mission_planner.launch.py start_encoder:=false
  ros2 launch gantry_control mission_planner.launch.py start_ic_cart:=false

Ctrl+C: gantry_controller disables motors and closes Teknic; dashboard calls /gantry/disable.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _payload_mount_defaults(config_dir):
    defaults = {
        'gantry_yaw_deg': '0.0',
        'gantry_pitch_deg': '0.0',
        'gantry_roll_deg': '0.0',
        'gantry_sign_x': '1.0',
        'gantry_sign_y': '1.0',
        'gantry_sign_z': '1.0',
    }
    path = os.path.join(config_dir, 'payload_mount.yaml')
    if not os.path.isfile(path):
        return defaults
    try:
        import yaml
        with open(path, encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        mount = data.get('payload_mount') or {}
        for key in defaults:
            if key in mount:
                defaults[key] = str(mount[key])
    except Exception:
        pass
    return defaults


def generate_launch_description():
    pkg_share = get_package_share_directory('gantry_control')
    gantry_launch = os.path.join(pkg_share, 'launch', 'gantry.launch.py')
    config_dir = os.path.join(pkg_share, 'config')
    fastdds_no_shm_xml = os.path.join(config_dir, 'fastdds_no_shm.xml')
    ic_cart_yaml = os.path.join(config_dir, 'ic_cart.yaml')
    mount = _payload_mount_defaults(config_dir)

    return LaunchDescription([
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', fastdds_no_shm_xml),
        SetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE', fastdds_no_shm_xml),

        DeclareLaunchArgument('joy_dev', default_value='/dev/input/js0'),
        DeclareLaunchArgument('start_gantry', default_value='true'),
        DeclareLaunchArgument('start_joy', default_value='true'),
        DeclareLaunchArgument('start_tracker', default_value='false'),
        DeclareLaunchArgument('start_phase1_tracker', default_value='false'),
        DeclareLaunchArgument('start_encoder', default_value='true'),
        DeclareLaunchArgument('encoder_backend', default_value='serial'),
        DeclareLaunchArgument('encoder_publish_rate_hz', default_value='100.0'),
        DeclareLaunchArgument('start_logger', default_value='false'),
        DeclareLaunchArgument('start_ic_cart', default_value='false'),
        DeclareLaunchArgument('ic_cart_port', default_value='/dev/ttyCH341USB1'),
        DeclareLaunchArgument('oak_ip', default_value='192.168.0.153'),
        DeclareLaunchArgument('marker_size', default_value='0.10'),
        DeclareLaunchArgument('stream_port', default_value='8092'),
        DeclareLaunchArgument('dashboard_port', default_value='8081'),
        DeclareLaunchArgument('workspace_m', default_value='1.15'),
        DeclareLaunchArgument('stack_pose_publish_hz', default_value='100.0'),
        DeclareLaunchArgument('stack_pose_sync_adaptive', default_value='false'),
        DeclareLaunchArgument('teknic_baud_rate', default_value='230400'),
        DeclareLaunchArgument('cart_velocity_source', default_value='measured'),
        DeclareLaunchArgument('motor_velocity_read_every_n', default_value='2'),
        DeclareLaunchArgument('position_velocity_alpha', default_value='0.25'),
        DeclareLaunchArgument('traj_write_on_change_only', default_value='false'),
        DeclareLaunchArgument('traj_write_keepalive_s', default_value='0.10'),
        DeclareLaunchArgument('traj_write_deadband_mm_s', default_value='0.5'),
        DeclareLaunchArgument('tracker_width', default_value='640'),
        DeclareLaunchArgument('tracker_height', default_value='400'),
        DeclareLaunchArgument('tracker_fps', default_value='128'),
        DeclareLaunchArgument('tracker_pos_alpha', default_value='0.40'),
        DeclareLaunchArgument('tracker_quad_decimate', default_value='2.0'),
        DeclareLaunchArgument('tracker_turbo', default_value='false'),
        DeclareLaunchArgument('tracker_camera_jpeg_fps', default_value='15.0'),
        DeclareLaunchArgument('tracker_decision_margin_min', default_value='8.0'),
        DeclareLaunchArgument('tracker_publish_alpha', default_value='0.22'),
        DeclareLaunchArgument('tracker_gantry_publish_alpha', default_value='0.24'),
        DeclareLaunchArgument('tracker_publish_max_step_m', default_value='0.008'),
        DeclareLaunchArgument('tracker_publish_median', default_value='5'),
        DeclareLaunchArgument('tracker_rigid_reproj_max_px', default_value='8.0'),
        DeclareLaunchArgument('gantry_yaw_deg', default_value=mount['gantry_yaw_deg']),
        DeclareLaunchArgument('gantry_pitch_deg', default_value=mount['gantry_pitch_deg']),
        DeclareLaunchArgument('gantry_roll_deg', default_value=mount['gantry_roll_deg']),
        DeclareLaunchArgument('gantry_sign_x', default_value=mount['gantry_sign_x']),
        DeclareLaunchArgument('gantry_sign_y', default_value=mount['gantry_sign_y']),
        DeclareLaunchArgument('gantry_sign_z', default_value=mount['gantry_sign_z']),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gantry_launch),
            launch_arguments={
                'joy_dev': LaunchConfiguration('joy_dev'),
                'start_gantry': LaunchConfiguration('start_gantry'),
                'start_joy': LaunchConfiguration('start_joy'),
                'start_tracker': LaunchConfiguration('start_tracker'),
                'start_phase1_tracker': LaunchConfiguration('start_phase1_tracker'),
                'start_encoder': LaunchConfiguration('start_encoder'),
                'encoder_backend': LaunchConfiguration('encoder_backend'),
                'encoder_publish_rate_hz': LaunchConfiguration('encoder_publish_rate_hz'),
                'start_logger': LaunchConfiguration('start_logger'),
                'oak_ip': LaunchConfiguration('oak_ip'),
                'marker_size': LaunchConfiguration('marker_size'),
                'stream_port': LaunchConfiguration('stream_port'),
                'stack_pose_publish_hz': LaunchConfiguration('stack_pose_publish_hz'),
                'stack_pose_sync_adaptive': LaunchConfiguration('stack_pose_sync_adaptive'),
                'teknic_baud_rate': LaunchConfiguration('teknic_baud_rate'),
                'cart_velocity_source': LaunchConfiguration('cart_velocity_source'),
                'motor_velocity_read_every_n': LaunchConfiguration('motor_velocity_read_every_n'),
                'position_velocity_alpha': LaunchConfiguration('position_velocity_alpha'),
                'traj_write_on_change_only': LaunchConfiguration('traj_write_on_change_only'),
                'traj_write_keepalive_s': LaunchConfiguration('traj_write_keepalive_s'),
                'traj_write_deadband_mm_s': LaunchConfiguration('traj_write_deadband_mm_s'),
                'tracker_width': LaunchConfiguration('tracker_width'),
                'tracker_height': LaunchConfiguration('tracker_height'),
                'tracker_fps': LaunchConfiguration('tracker_fps'),
                'tracker_pos_alpha': LaunchConfiguration('tracker_pos_alpha'),
                'tracker_quad_decimate': LaunchConfiguration('tracker_quad_decimate'),
                'tracker_turbo': LaunchConfiguration('tracker_turbo'),
                'tracker_camera_jpeg_fps': LaunchConfiguration(
                    'tracker_camera_jpeg_fps'),
                'tracker_decision_margin_min': LaunchConfiguration(
                    'tracker_decision_margin_min'),
                'tracker_publish_alpha': LaunchConfiguration('tracker_publish_alpha'),
                'tracker_gantry_publish_alpha': LaunchConfiguration(
                    'tracker_gantry_publish_alpha'),
                'tracker_publish_max_step_m': LaunchConfiguration(
                    'tracker_publish_max_step_m'),
                'tracker_publish_median': LaunchConfiguration('tracker_publish_median'),
                'tracker_rigid_reproj_max_px': LaunchConfiguration(
                    'tracker_rigid_reproj_max_px'),
                'gantry_yaw_deg': LaunchConfiguration('gantry_yaw_deg'),
                'gantry_pitch_deg': LaunchConfiguration('gantry_pitch_deg'),
                'gantry_roll_deg': LaunchConfiguration('gantry_roll_deg'),
                'gantry_sign_x': LaunchConfiguration('gantry_sign_x'),
                'gantry_sign_y': LaunchConfiguration('gantry_sign_y'),
                'gantry_sign_z': LaunchConfiguration('gantry_sign_z'),
            }.items(),
        ),

        Node(
            package='gantry_control',
            executable='ic_cart_node.py',
            name='ic_cart_node',
            output='screen',
            parameters=[
                ic_cart_yaml,
                {'serial_port': LaunchConfiguration('ic_cart_port')},
            ],
            condition=IfCondition(LaunchConfiguration('start_ic_cart')),
        ),

        Node(
            package='gantry_control',
            executable='crane_dashboard_server.py',
            name='crane_dashboard_server',
            output='screen',
            parameters=[{
                'dashboard_port': LaunchConfiguration('dashboard_port'),
                'camera_proxy': True,
                'stream_port': LaunchConfiguration('stream_port'),
                'workspace_m': LaunchConfiguration('workspace_m'),
                'stack_pose_publish_hz': LaunchConfiguration('stack_pose_publish_hz'),
            }],
        ),
    ])
