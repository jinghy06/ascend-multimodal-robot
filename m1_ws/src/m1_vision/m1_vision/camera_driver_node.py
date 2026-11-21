"""camera_driver_node：USB 相机图像采集（v4l2）。

发布 /camera/image_raw 与 /camera/camera_info。
支持 video_file 参数循环播放视频文件，用于无相机调试（W2 前的 PC/板上开发）。
"""

import os
import time

import cv2
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from m1_vision.ros_image import cv2_to_imgmsg


class CameraDriverNode(Node):
    def __init__(self):
        super().__init__("camera_driver_node")
        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("video_file", "")       # 非空则循环播放该视频（调试模式）
        self.declare_parameter("width", 1920)
        self.declare_parameter("height", 1080)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("camera_info_yaml", "")  # ROS 标定 yaml，缺省用内参近似值
        self.declare_parameter("fx", 1200.0)
        self.declare_parameter("fy", 1200.0)
        self.declare_parameter("object_image_topic", "")
        self.declare_parameter("object_publish_fps", 2.0)

        self.frame_id = self.get_parameter("frame_id").value
        width = self.get_parameter("width").value
        height = self.get_parameter("height").value
        fps = self.get_parameter("fps").value
        video_file = self.get_parameter("video_file").value

        if video_file:
            if not os.path.exists(video_file):
                raise RuntimeError(f"video_file 不存在: {video_file}")
            self.cap = cv2.VideoCapture(video_file)
            self.video_file = video_file
            self.get_logger().warn(f"调试模式：循环播放视频 {video_file}")
        else:
            self.cap = cv2.VideoCapture(self.get_parameter("device").value, cv2.CAP_V4L2)
            self.video_file = ""
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError("相机/视频打开失败")

        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.img_pub = self.create_publisher(Image, "/camera/image_raw", qos)
        object_topic = self.get_parameter("object_image_topic").value
        self.object_img_pub = self.create_publisher(Image, object_topic, qos) if object_topic else None
        self.object_period = 1.0 / self.get_parameter("object_publish_fps").value
        self._last_object_publish = 0.0
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 5)
        self.camera_info = self._load_camera_info(width, height)
        self._info_from_calibration = bool(
            self.get_parameter("camera_info_yaml").value
            and os.path.exists(self.get_parameter("camera_info_yaml").value))

        self.timer = self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(
            f"camera_driver_node 启动：{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {fps} FPS")

    def _load_camera_info(self, width, height):
        path = self.get_parameter("camera_info_yaml").value
        info = CameraInfo()
        info.header.frame_id = self.frame_id
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                calib = yaml.safe_load(f)
            info.width = calib["image_width"]
            info.height = calib["image_height"]
            info.k = [float(v) for v in calib["camera_matrix"]["data"]]
            info.d = [float(v) for v in calib["distortion_coefficients"]["data"]]
            info.r = [float(v) for v in calib["rectification_matrix"]["data"]]
            info.p = [float(v) for v in calib["projection_matrix"]["data"]]
            info.distortion_model = calib.get("distortion_model", "plumb_bob")
            self.get_logger().info(f"已加载相机标定文件 {path}")
        else:
            # 未标定前的近似内参（W2 标定后务必改用 camera_info_yaml）
            fx = self.get_parameter("fx").value
            fy = self.get_parameter("fy").value
            info.width, info.height = width, height
            info.k = [fx, 0.0, width / 2.0, 0.0, fy, height / 2.0, 0.0, 0.0, 1.0]
            info.d = [0.0] * 5
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, width / 2.0, 0.0, 0.0, fy, height / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            info.distortion_model = "plumb_bob"
            self.get_logger().warn("未提供标定文件，使用近似内参，测距精度不保证（W2 标定后替换）")
        return info

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.video_file:  # 循环播放
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            self.get_logger().warn("相机读帧失败", throttle_duration_sec=2.0)
            return
        stamp = self.get_clock().now().to_msg()
        # 未标定时按实际帧尺寸更新内参（视频文件/相机实际分辨率可能与参数不一致）
        if not self._info_from_calibration:
            h, w = frame.shape[:2]
            if w != self.camera_info.width or h != self.camera_info.height:
                fx = self.get_parameter("fx").value * w / self.camera_info.width
                fy = self.get_parameter("fy").value * h / self.camera_info.height
                self.camera_info.width, self.camera_info.height = w, h
                self.camera_info.k = [fx, 0.0, w / 2.0, 0.0, fy, h / 2.0, 0.0, 0.0, 1.0]
                self.camera_info.p = [fx, 0.0, w / 2.0, 0.0, 0.0, fy, h / 2.0, 0.0,
                                      0.0, 0.0, 1.0, 0.0]
                self.get_logger().info(
                    f"按实际帧尺寸更新内参：{w}x{h}, fx={fx:.1f}", once=True)
        img_msg = cv2_to_imgmsg(frame, "bgr8")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.frame_id
        self.camera_info.header.stamp = stamp
        self.img_pub.publish(img_msg)
        if self.object_img_pub is not None:
            now = time.monotonic()
            if now - self._last_object_publish >= self.object_period:
                self.object_img_pub.publish(img_msg)
                self._last_object_publish = now
        self.info_pub.publish(self.camera_info)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
