"""mock_perception_node：无硬件联调工具。

在没有相机/雷达/NPU 的开发机上发布仿真数据，用于打通
fusion_ranging_node → target_tracking_node 链路（W7 前在 PC 上验证状态机）：
  /camera/camera_info      合成内参
  /scan                    一堵 3 m 正墙 + 一个绕圈移动的"人腿"簇
  /m1/person_detections    与仿真簇方位一致的合成人体框
  /m1/object_detections    合成杯子检测框（vision_msgs）
  /m1/laser_spot           固定方位的合成光斑（每 10 s 出现 3 s）

用法：ros2 run m1_vision mock_perception_node
然后另开终端启动 fusion_ranging_node / target_tracking_node 观察输出。
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, LaserScan
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from m1_msgs.msg import PersonDetection, PersonDetectionArray

FX, CX, IMG_W, IMG_H = 1200.0, 960.0, 1920, 1080


class MockPerceptionNode(Node):
    def __init__(self):
        super().__init__("mock_perception_node")
        self.t = 0.0
        self.create_timer(0.05, self._tick)  # 20 Hz
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 5)
        self.det_pub = self.create_publisher(
            PersonDetectionArray, "/m1/person_detections", 10)
        self.spot_pub = self.create_publisher(PointStamped, "/m1/laser_spot", 10)
        self.object_pub = self.create_publisher(
            Detection2DArray, "/m1/object_detections", 10)
        self.get_logger().info("mock_perception_node 启动（仿真数据，仅供无硬件联调）")

    def _tick(self):
        self.t += 0.05
        stamp = self.get_clock().now().to_msg()

        # 仿真人体：方位角在 ±0.5 rad 内缓慢摆动，距离 1.2-2.0 m 呼吸变化
        theta_p = 0.5 * math.sin(self.t * 0.4)
        dist_p = 1.6 + 0.4 * math.sin(self.t * 0.25)

        # ---- /scan：720 点，[-pi, pi]，正墙 3 m，人腿簇 0.06 rad 宽 ----
        n = 720
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = "laser_frame"
        scan.angle_min, scan.angle_max = -math.pi, math.pi
        scan.angle_increment = 2 * math.pi / n
        scan.range_min, scan.range_max = 0.12, 8.0
        angles = -math.pi + np.arange(n) * scan.angle_increment
        ranges = np.full(n, 3.0 / np.maximum(np.cos(np.clip(angles, -1.4, 1.4)), 0.2))
        person_hit = np.abs(np.arctan2(np.sin(angles - theta_p),
                                       np.cos(angles - theta_p))) < 0.03
        ranges[person_hit] = dist_p
        scan.ranges = ranges.tolist()
        self.scan_pub.publish(scan)

        # ---- camera_info ----
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "camera_link"
        info.width, info.height = IMG_W, IMG_H
        info.k = [FX, 0.0, CX, 0.0, FX, IMG_H / 2.0, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [FX, 0.0, CX, 0.0, 0.0, FX, IMG_H / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = "plumb_bob"
        self.info_pub.publish(info)

        # ---- 合成人体检测（假设 camera 与 laser 同轴，TF 兜底 yaw=0）----
        dets = PersonDetectionArray()
        dets.header.stamp = stamp
        dets.header.frame_id = "camera_link"
        det = PersonDetection()
        u = CX + FX * math.tan(theta_p)
        det.x1, det.x2 = u - 120.0, u + 120.0
        det.y1, det.y2 = 200.0, 980.0
        det.score = 0.85
        dets.detections.append(det)
        self.det_pub.publish(dets)

        objects = Detection2DArray()
        objects.header = dets.header
        cup = Detection2D()
        cup.header = dets.header
        cup.bbox.center.position.x = CX + FX * math.tan(0.2)
        cup.bbox.center.position.y = IMG_H / 2.0
        cup.bbox.size_x = cup.bbox.size_y = 100.0
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = "cup"
        hypothesis.hypothesis.score = 0.9
        hypothesis.pose.pose.orientation.w = 1.0
        cup.results.append(hypothesis)
        objects.detections.append(cup)
        self.object_pub.publish(objects)

        # ---- 合成光斑：每 10 s 出现 3 s，固定指向右前方 1.2 m 墙面 ----
        spot = PointStamped()
        spot.header.stamp = stamp
        spot.header.frame_id = "camera_link"
        if (self.t % 10.0) < 3.0:
            theta_s = -0.4
            spot.point.x = CX + FX * math.tan(theta_s)
            spot.point.y = 540.0
            spot.point.z = 0.9
        else:
            spot.point.z = -1.0
        self.spot_pub.publish(spot)


def main(args=None):
    rclpy.init(args=args)
    node = MockPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
