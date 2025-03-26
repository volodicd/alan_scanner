from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    movement_controller_node = Node(
        package='turtlebot_movement_test',
        executable='movement_controller',
        name='movement_controller',
        output='screen',
        parameters=[{
            'obstacle_threshold': 0.3,   # still useful if you have a /scan in the future
            'linear_speed': 0.2,         # 20 cm/s
            'angular_speed': 0.5,        # 0.5 rad/s
            'update_rate': 10.0,         # main control loop in Hz
            'cmd_rate': 5.0              # velocity publish rate in Hz
        }]
    )

    return LaunchDescription([movement_controller_node])
