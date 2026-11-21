"""Atlas 处理器侧统一入口：M1 视觉与 M2 语音，支持实机/仿真。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    m1_share = get_package_share_directory("m1_vision")
    params_file = LaunchConfiguration("params_file")
    voice_root = LaunchConfiguration("voice_root")
    use_mock = LaunchConfiguration("use_mock")
    enable_asr = LaunchConfiguration("enable_asr")
    enable_tts = LaunchConfiguration("enable_tts")
    enable_semantic = LaunchConfiguration("enable_semantic")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=os.path.join(
            m1_share, "config", "m1_vision.yaml")),
        DeclareLaunchArgument("voice_root", default_value="/root/ascend-multimodal-robot/voice_control"),
        DeclareLaunchArgument("use_mock", default_value="false"),
        DeclareLaunchArgument("enable_asr", default_value="true"),
        DeclareLaunchArgument("enable_tts", default_value="false"),
        DeclareLaunchArgument("enable_semantic", default_value="false"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(m1_share, "launch", "m1_mock.launch.py")),
            condition=IfCondition(use_mock)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(m1_share, "launch", "m1_bringup.launch.py")),
            launch_arguments={"params_file": params_file}.items(),
            condition=UnlessCondition(use_mock)),
        Node(package="voice_control", executable="parser_node", output="screen"),
        Node(package="voice_control", executable="asr_node", output="screen",
             condition=IfCondition(enable_asr),
             parameters=[{
                 "kws_dir": [voice_root, "/models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile"],
                 "asr_dir": [voice_root, "/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"],
                 "vad_model": [voice_root, "/models/silero_vad_v4.onnx"],
                 "keywords_file": [voice_root, "/configs/keywords.txt"],
             }]),
        Node(package="voice_control", executable="tts_node", output="screen",
             condition=IfCondition(enable_tts),
             parameters=[{"tts_yaml": [voice_root, "/configs/tts.yaml"]}]),
        Node(package="voice_control", executable="llm_service", output="screen",
             condition=IfCondition(enable_semantic),
             parameters=[{"llm_yaml": [voice_root, "/configs/llm.yaml"]}]),
    ])
