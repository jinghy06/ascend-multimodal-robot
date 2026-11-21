"""无传感器 M1 全链路仿真：人体、物体、激光光斑、测距、跟踪与导航目标。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory("m1_vision"), "config", "m1_vision.yaml")
    return LaunchDescription([
        Node(package="m1_vision", executable="mock_perception_node", output="screen"),
        Node(package="m1_vision", executable="fusion_ranging_node",
             parameters=[params], output="screen"),
        Node(package="m1_vision", executable="target_tracking_node",
             parameters=[params], output="screen"),
        Node(package="tf2_ros", executable="static_transform_publisher",
             arguments=["0", "0", "0", "0", "0", "0", "1", "map", "base_link"],
             output="screen"),
        Node(package="tf2_ros", executable="static_transform_publisher",
             arguments=["0", "0", "0", "0", "0", "0", "1", "base_link", "laser_frame"],
             output="screen"),
    ])
