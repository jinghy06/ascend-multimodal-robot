import os
from glob import glob

from setuptools import find_packages, setup

package_name = "m1_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.xacro")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="M1 Owner",
    maintainer_email="you@example.com",
    description="M1 视觉感知模块（Atlas 200I DK A2）",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_driver_node = m1_vision.camera_driver_node:main",
            "yolo_npu_node = m1_vision.yolo_npu_node:main",
            "object_detection_node = m1_vision.object_detection_node:main",
            "laser_spot_node = m1_vision.laser_spot_node:main",
            "fusion_ranging_node = m1_vision.fusion_ranging_node:main",
            "target_tracking_node = m1_vision.target_tracking_node:main",
            "mock_perception_node = m1_vision.mock_perception_node:main",
        ],
    },
)
