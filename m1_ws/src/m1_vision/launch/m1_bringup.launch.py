"""M1 视觉感知模块一键启动（6 个节点）。

用法：
  ros2 launch m1_vision m1_bringup.launch.py
  ros2 launch m1_vision m1_bringup.launch.py params_file:=/path/to/other.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory("m1_vision"), "config", "m1_vision.yaml")
    params_file = LaunchConfiguration("params_file")

    nodes = [
        "camera_driver_node",
        "yolo_npu_node",
        "object_detection_node",
        "laser_spot_node",
        "fusion_ranging_node",
        "target_tracking_node",
    ]
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        *[Node(package="m1_vision", executable=name, name=name,
               parameters=[params_file], output="screen") for name in nodes],
    ])
