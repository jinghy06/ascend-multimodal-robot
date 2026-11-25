"""fusion_ranging_node：相机-雷达融合测距（设计文档 5.2 / 5.4 节）。

订阅人体检测框与光斑像素坐标，经相机内参反投影为视线方向、TF 外参转到
laser_frame 得到方位角 θ；在 /scan 的 θ±window 窗口内做一维聚类，取最近
有效簇的中位距离（抗单点噪声，人腿 1-2 个小簇场景）。

输出：
  /m1/ranged_targets      m1_msgs/RangedTargetArray（人体，极坐标，laser_frame）
  /m1/laser_spot_ranged   geometry_msgs/PointStamped（光斑，直角坐标，laser_frame）
  /m1/grasp_target        vision_msgs/Detection2DArray（物体类别、像素框及测距位置）
"""

import copy
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray

from m1_msgs.msg import PersonDetectionArray, RangedTarget, RangedTargetArray
from m1_vision.tf_utils import quat_rotate


class FusionRangingNode(Node):
    def __init__(self):
        super().__init__("fusion_ranging_node")
        self.declare_parameter("laser_frame", "laser_frame")
        self.declare_parameter("window_deg", 5.0)         # 方位角窗口半宽
        self.declare_parameter("cluster_gap", 0.15)       # 相邻点距离差 > 该值则分簇
        self.declare_parameter("min_cluster_points", 3)
        self.declare_parameter("fallback_hfov_deg", 78.0)  # camera_info 未到时的近似水平视场角
        self.declare_parameter("fallback_mount_yaw_deg", 0.0)  # TF 不可用时的安装偏航角兜底
        self.declare_parameter("image_width_fallback", 1920)

        self.laser_frame = self.get_parameter("laser_frame").value
        self.window_rad = math.radians(self.get_parameter("window_deg").value)
        self.cluster_gap = self.get_parameter("cluster_gap").value
        self.min_cluster_points = self.get_parameter("min_cluster_points").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._tf_warned = False

        self.fx = None      # 相机内参，等 camera_info
        self.cx = None
        self.latest_scan = None

        self.ranged_pub = self.create_publisher(
            RangedTargetArray, "/m1/ranged_targets", 10)
        self.spot_ranged_pub = self.create_publisher(
            PointStamped, "/m1/laser_spot_ranged", 10)
        self.grasp_pub = self.create_publisher(
            Detection2DArray, "/m1/grasp_target", 10)

        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(LaserScan, "/scan", self._on_scan, sensor_qos)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, 5)
        self.create_subscription(PersonDetectionArray, "/m1/person_detections",
                                 self._on_detections, 10)
        self.create_subscription(PointStamped, "/m1/laser_spot",
                                 self._on_spot, 10)
        self.create_subscription(Detection2DArray, "/m1/object_detections",
                                 self._on_objects, 10)
        self.get_logger().info("fusion_ranging_node 启动")

    def _on_info(self, msg: CameraInfo):
        if self.fx is None:
            self.fx, self.cx = msg.k[0], msg.k[2]
            self.get_logger().info(f"相机内参就绪：fx={self.fx:.1f}, cx={self.cx:.1f}")

    def _on_scan(self, msg: LaserScan):
        self.latest_scan = msg

    # ---- 方位角解算 ----
    def _bearing(self, u):
        """像素横坐标 u -> laser_frame 方位角 rad（正前为 0，左正）。"""
        if self.fx is not None:
            x_cam = (u - self.cx) / self.fx
        else:  # 标定前用水平视场角线性近似
            w = self.get_parameter("image_width_fallback").value
            hfov = math.radians(self.get_parameter("fallback_hfov_deg").value)
            x_cam = (u - w / 2.0) / (w / 2.0) * math.tan(hfov / 2.0)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.laser_frame, "camera_link", Time())
            q = tf.transform.rotation
            ray = quat_rotate((q.x, q.y, q.z, q.w), (x_cam, 0.0, 1.0))
            return math.atan2(ray[1], ray[0])
        except TransformException:
            if not self._tf_warned:
                self.get_logger().warn(
                    "camera_link->laser_frame TF 不可用，使用安装偏航角兜底参数")
                self._tf_warned = True
            yaw = math.radians(self.get_parameter("fallback_mount_yaw_deg").value)
            return math.atan(x_cam) + yaw

    # ---- 雷达窗口聚类测距 ----
    def _range_at(self, theta):
        """在 theta±window 内聚类，返回最近有效簇的中位距离；无有效点返回 None。"""
        scan = self.latest_scan
        if scan is None:
            return None
        angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        # scan 角度可能不在 [-pi, pi]，用相位差取窗
        dphi = np.abs(np.arctan2(np.sin(angles - theta), np.cos(angles - theta)))
        ranges = np.asarray(scan.ranges)
        sel = (dphi <= self.window_rad) & np.isfinite(ranges) \
            & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        if not np.any(sel):
            return None
        pts = np.sort(ranges[sel])
        # 按相邻距离差分簇
        splits = np.where(np.diff(pts) > self.cluster_gap)[0] + 1
        clusters = [c for c in np.split(pts, splits) if len(c) >= self.min_cluster_points]
        if not clusters:
            return float(np.min(pts))  # 点数不足时退化为最近点
        nearest = min(clusters, key=lambda c: np.median(c))
        return float(np.median(nearest))

    # ---- 回调 ----
    def _on_detections(self, msg: PersonDetectionArray):
        out = RangedTargetArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.laser_frame
        for det in msg.detections:
            theta = self._bearing((det.x1 + det.x2) / 2.0)
            dist = self._range_at(theta)
            if dist is None:
                continue
            t = RangedTarget()
            t.theta, t.distance, t.score = theta, dist, det.score
            t.source = RangedTarget.SOURCE_PERSON
            out.targets.append(t)
        self.ranged_pub.publish(out)

    def _on_objects(self, msg: Detection2DArray):
        out = Detection2DArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.laser_frame
        for det in msg.detections:
            theta = self._bearing(det.bbox.center.position.x)
            dist = self._range_at(theta)
            if dist is None:
                continue
            ranged = copy.deepcopy(det)
            ranged.header.frame_id = self.laser_frame
            for result in ranged.results:
                result.pose.pose.position.x = dist * math.cos(theta)
                result.pose.pose.position.y = dist * math.sin(theta)
                result.pose.pose.position.z = 0.0
                result.pose.pose.orientation.w = 1.0
            out.detections.append(ranged)
        self.grasp_pub.publish(out)

    def _on_spot(self, msg: PointStamped):
        if msg.point.z < 0.0:
            return  # 未检出帧
        theta = self._bearing(msg.point.x)
        dist = self._range_at(theta)
        if dist is None:
            self.get_logger().warn("光斑方位无有效雷达回波", throttle_duration_sec=2.0)
            return
        out = PointStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.laser_frame
        out.point.x = dist * math.cos(theta)
        out.point.y = dist * math.sin(theta)
        out.point.z = msg.point.z  # 透传强度，供跟踪节点参考
        self.spot_ranged_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FusionRangingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
