"""
Minimal stack: gantry controller + traj_cmd logger only.

  ros2 launch gantry_control traj_log.launch.py

Then (other terminals):
  ros2 run gantry_control traj_player.py ~/csv_profiles/pulse.csv
  ros2 run gantry_control gantry_cli.py enable
"""
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
import os


def generate_launch_description():
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.normpath(os.path.join(launch_dir, '..', 'config'))
    experiment_yaml = os.path.join(config_dir, 'experiment.yaml')
    gantry_yaml = os.path.join(config_dir, 'gantry.yaml')

    return LaunchDescription([
        SetEnvironmentVariable('JETSON_MODEL_NAME', 'JETSON_ORIN_NANO'),

        Node(
            package='gantry_control',
            executable='gantry_controller',
            name='gantry_controller',
            output='screen',
            emulate_tty=True,
            parameters=[gantry_yaml],
        ),

        Node(
            package='experiment_logger',
            executable='logger_node',
            name='logger_node',
            output='screen',
            parameters=[experiment_yaml],
        ),
    ])
