"""
Launch gantry controller, joystick, optional payload tracker, encoder, and logger.

Packages (crane_ws):
  gantry_control      — crane + TRAJ
  payload_perception  — optional OAK tracker + payload encoders
  experiment_logger   — CSV logging

Default: encoder off (not plugged in). Enable with start_encoder:=true.

The multi-face phase1 tracker runs by default (start_phase1_tracker:=true).
Do NOT also pass start_tracker:=true — that is the legacy single-tag tracker
and the two fight over the one OAK-D camera. Pass the camera IP as oak_ip:

  ros2 launch gantry_control gantry.launch.py oak_ip:=192.168.0.153
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _payload_mount_defaults(config_dir):
    """Load saved tripod mount calibration from config/payload_mount.yaml."""
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
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.normpath(os.path.join(launch_dir, '..', 'config'))
    experiment_yaml = os.path.join(config_dir, 'experiment.yaml')
    encoder_yaml = os.path.join(config_dir, 'encoder.yaml')
    encoder_serial_yaml = os.path.join(config_dir, 'encoder_serial.yaml')
    fastdds_no_shm_xml = os.path.join(config_dir, 'fastdds_no_shm.xml')
    gantry_yaml = os.path.join(config_dir, 'gantry.yaml')
    mount = _payload_mount_defaults(config_dir)

    return LaunchDescription([
        SetEnvironmentVariable('JETSON_MODEL_NAME', 'JETSON_ORIN_NANO'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', fastdds_no_shm_xml),
        SetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE', fastdds_no_shm_xml),

        DeclareLaunchArgument('joy_dev', default_value='/dev/input/js0'),
        SetEnvironmentVariable(
            'SDL_JOYSTICK_DEVICE',
            LaunchConfiguration('joy_dev'),
        ),

        DeclareLaunchArgument('start_gantry', default_value='true'),
        DeclareLaunchArgument('start_joy', default_value='true'),
        DeclareLaunchArgument('start_tracker', default_value='false'),
        DeclareLaunchArgument('start_phase1_tracker', default_value='false'),
        DeclareLaunchArgument('start_encoder', default_value='false'),
        DeclareLaunchArgument('encoder_backend', default_value='serial'),
        DeclareLaunchArgument('encoder_publish_rate_hz', default_value='100.0'),
        DeclareLaunchArgument('start_gantry_frame', default_value='false'),
        DeclareLaunchArgument('start_logger', default_value='true'),
        DeclareLaunchArgument('oak_ip', default_value='192.168.0.153'),
        DeclareLaunchArgument('marker_size', default_value='0.10'),
        DeclareLaunchArgument('stream_port', default_value='8080'),
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
        DeclareLaunchArgument('tracker_sync_adaptive', default_value='true'),
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

        Node(
            package='gantry_control',
            executable='gantry_controller',
            name='gantry_controller',
            output='screen',
            emulate_tty=True,
            parameters=[
                gantry_yaml,
                {
                    'stack_pose_publish_hz': ParameterValue(
                        LaunchConfiguration('stack_pose_publish_hz'),
                        value_type=float,
                    ),
                    'stack_pose_sync_adaptive': ParameterValue(
                        LaunchConfiguration('stack_pose_sync_adaptive'),
                        value_type=bool,
                    ),
                    'teknic_baud_rate': ParameterValue(
                        LaunchConfiguration('teknic_baud_rate'),
                        value_type=int,
                    ),
                    'cart_velocity_source': LaunchConfiguration('cart_velocity_source'),
                    'motor_velocity_read_every_n': ParameterValue(
                        LaunchConfiguration('motor_velocity_read_every_n'),
                        value_type=int,
                    ),
                    'position_velocity_alpha': ParameterValue(
                        LaunchConfiguration('position_velocity_alpha'),
                        value_type=float,
                    ),
                    'traj_write_on_change_only': ParameterValue(
                        LaunchConfiguration('traj_write_on_change_only'),
                        value_type=bool,
                    ),
                    'traj_write_keepalive_s': ParameterValue(
                        LaunchConfiguration('traj_write_keepalive_s'),
                        value_type=float,
                    ),
                    'traj_write_deadband_mm_s': ParameterValue(
                        LaunchConfiguration('traj_write_deadband_mm_s'),
                        value_type=float,
                    ),
                },
            ],
            condition=IfCondition(LaunchConfiguration('start_gantry')),
        ),

        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'device_id': 0,
                'deadzone': 0.05,
                'autorepeat_rate': 50.0,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_joy')),
        ),

        Node(
            package='payload_perception',
            executable='encoder_node',
            name='encoder_node',
            output='screen',
            parameters=[encoder_yaml],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('start_encoder'), "' == 'true' and '",
                LaunchConfiguration('encoder_backend'), "' == 'gpio'",
            ])),
        ),

        Node(
            package='payload_perception',
            executable='encoder_serial_node',
            name='encoder_serial_node',
            output='screen',
            parameters=[
                encoder_serial_yaml,
                {
                    'publish_rate_hz': ParameterValue(
                        LaunchConfiguration('encoder_publish_rate_hz'),
                        value_type=float,
                    ),
                },
            ],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('start_encoder'), "' == 'true' and '",
                LaunchConfiguration('encoder_backend'), "' == 'serial'",
            ])),
        ),

        Node(
            package='payload_perception',
            executable='payload_tracker',
            name='payload_tracker',
            output='screen',
            arguments=[
                '--ip', LaunchConfiguration('oak_ip'),
                '--width', LaunchConfiguration('tracker_width'),
                '--height', LaunchConfiguration('tracker_height'),
                '--fps', LaunchConfiguration('tracker_fps'),
                '--marker-size', LaunchConfiguration('marker_size'),
                '--quad-decimate', LaunchConfiguration('tracker_quad_decimate'),
                '--decision-margin-min', LaunchConfiguration('tracker_decision_margin_min'),
                '--ros',
                '--no-wait-motion',
                '--stream-port', LaunchConfiguration('stream_port'),
                '--stream-detect-only',
                '--ros-publish-hz', LaunchConfiguration('stack_pose_publish_hz'),
                '--camera-jpeg-fps', LaunchConfiguration('tracker_camera_jpeg_fps'),
                '--sync-adaptive',
                '--publish-alpha', LaunchConfiguration('tracker_publish_alpha'),
                '--gantry-publish-alpha',
                LaunchConfiguration('tracker_gantry_publish_alpha'),
                '--publish-max-step-m',
                LaunchConfiguration('tracker_publish_max_step_m'),
                '--publish-median', LaunchConfiguration('tracker_publish_median'),
                '--rigid-reproj-max-px',
                LaunchConfiguration('tracker_rigid_reproj_max_px'),
                '--pos-alpha', LaunchConfiguration('tracker_pos_alpha'),
                '--gantry-yaw-deg', LaunchConfiguration('gantry_yaw_deg'),
                '--gantry-pitch-deg', LaunchConfiguration('gantry_pitch_deg'),
                '--gantry-roll-deg', LaunchConfiguration('gantry_roll_deg'),
                '--gantry-sign-x', LaunchConfiguration('gantry_sign_x'),
                '--gantry-sign-y', LaunchConfiguration('gantry_sign_y'),
                '--gantry-sign-z', LaunchConfiguration('gantry_sign_z'),
            ],
            # Legacy single-tag-pair tracker.  Mutually exclusive with the
            # phase1 multi-face tracker below: both are named 'payload_tracker',
            # both open the single OAK-D camera, and both publish /payload/state,
            # so running both makes them race for the camera and interleave
            # garbage.  phase1 wins by default; the legacy node only starts when
            # start_tracker=true AND start_phase1_tracker=false.
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('start_tracker'), "' == 'true' and '",
                LaunchConfiguration('start_phase1_tracker'), "' == 'false'",
            ])),
        ),

        Node(
            package='payload_perception',
            executable='phase1_tracker_node',
            name='payload_tracker',
            output='screen',
            # phase1_tracker_node takes no CLI args; it reads the OAK-D IP from
            # face_config.yaml camera.ip or the OAK_IP env var.  Forward the
            # launch oak_ip so it can run standalone (empty = auto-discover).
            additional_env={'OAK_IP': LaunchConfiguration('oak_ip')},
            condition=IfCondition(LaunchConfiguration('start_phase1_tracker')),
        ),

        Node(
            package='payload_perception',
            executable='payload_gantry_frame',
            name='payload_gantry_frame',
            output='screen',
            parameters=[experiment_yaml],
            condition=IfCondition(LaunchConfiguration('start_gantry_frame')),
        ),

        Node(
            package='experiment_logger',
            executable='logger_node',
            name='logger_node',
            output='screen',
            parameters=[experiment_yaml],
            condition=IfCondition(LaunchConfiguration('start_logger')),
        ),
    ])
