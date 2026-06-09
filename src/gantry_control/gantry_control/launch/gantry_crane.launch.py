"""
Crane experiment launch: gantry + joy + logger, encoder off.
Camera payload tracking is opt-in.

  ros2 launch gantry_control gantry_crane.launch.py
  ros2 launch gantry_control gantry_crane.launch.py start_tracker:=true oak_ip:=192.168.0.153
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('gantry_control')
    gantry_launch = os.path.join(pkg_share, 'launch', 'gantry.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument('start_gantry', default_value='true'),
        DeclareLaunchArgument('start_joy', default_value='true'),
        DeclareLaunchArgument('start_tracker', default_value='false'),
        DeclareLaunchArgument('start_logger', default_value='true'),
        DeclareLaunchArgument('oak_ip', default_value='192.168.0.153'),
        DeclareLaunchArgument('marker_size', default_value='0.10'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gantry_launch),
            launch_arguments={
                'start_encoder': 'false',
                'start_gantry': LaunchConfiguration('start_gantry'),
                'start_joy': LaunchConfiguration('start_joy'),
                'start_tracker': LaunchConfiguration('start_tracker'),
                'start_logger': LaunchConfiguration('start_logger'),
                'oak_ip': LaunchConfiguration('oak_ip'),
                'marker_size': LaunchConfiguration('marker_size'),
            }.items(),
        ),
    ])
