from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('hmis'),
        'config',
        'hmis.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config),
        Node(
            package='hmis',
            executable='hmis_node',
            name='hmis',
            output='screen',
            emulate_tty=True,
            parameters=[LaunchConfiguration('config')],
        ),
    ])
