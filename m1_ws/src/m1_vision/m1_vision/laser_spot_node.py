"""laser_spot_node：激光笔光斑检测（HSV 阈值 + 形态学 + 连通域，CPU/OpenCV）。

算法对应设计文档 5.3 节：
BGR→HSV 颜色分割（红激光 H 双区间）→ 3x3 开/闭运算 → 连通域筛选
（面积窗 + 圆形度 4πA/P² > circularity_min）→ V 通道亮度加权亚像素质心。

发布 /m1/laser_spot（geometry_msgs/PointStamped，x=u, y=v 像素，z=归一化强度，
z<0 表示本帧未检出）与调试图像 /m1/laser_spot/debug。
"""

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from m1_vision.ros_image import cv2_to_imgmsg, imgmsg_to_cv2


class LaserSpotNode(Node):
    def __init__(self):
        super().__init__("laser_spot_node")
        self.declare_parameter("laser_color", "red")      # red / green
        # 红激光：H 近 0 与近 180 两个区间
        self.declare_parameter("red_h_low1", 0)
        self.declare_parameter("red_h_high1", 10)
        self.declare_parameter("red_h_low2", 170)
        self.declare_parameter("red_h_high2", 180)
        self.declare_parameter("green_h_low", 35)
        self.declare_parameter("green_h_high", 90)
        self.declare_parameter("sat_min", 120)            # 饱和度下限，滤环境光
        self.declare_parameter("val_min", 150)            # 亮度下限
        self.declare_parameter("area_min", 5.0)
        self.declare_parameter("area_max", 2000.0)
        self.declare_parameter("circularity_min", 0.6)
        self.declare_parameter("morph_kernel", 3)
        self.declare_parameter("publish_debug_image", True)

        self.spot_pub = self.create_publisher(PointStamped, "/m1/laser_spot", 10)
        self.debug_pub = self.create_publisher(Image, "/m1/laser_spot/debug", 2)

        qos = QoSProfile(depth=2)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self._on_image, qos)
        k = self.get_parameter("morph_kernel").value
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.get_logger().info(
            f"laser_spot_node 启动，激光颜色：{self.get_parameter('laser_color').value}")

    def _hsv_mask(self, hsv):
        p = self.get_parameter
        sat_min, val_min = p("sat_min").value, p("val_min").value
        if p("laser_color").value == "green":
            mask = cv2.inRange(hsv,
                               np.array([p("green_h_low").value, sat_min, val_min]),
                               np.array([p("green_h_high").value, 255, 255]))
        else:
            m1 = cv2.inRange(hsv,
                             np.array([p("red_h_low1").value, sat_min, val_min]),
                             np.array([p("red_h_high1").value, 255, 255]))
            m2 = cv2.inRange(hsv,
                             np.array([p("red_h_low2").value, sat_min, val_min]),
                             np.array([p("red_h_high2").value, 255, 255]))
            mask = cv2.bitwise_or(m1, m2)
        return mask

    def _on_image(self, msg: Image):
        frame = imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._hsv_mask(hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)   # 去噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)  # 补空洞

        spot = self._pick_spot(mask, hsv[:, :, 2])

        out = PointStamped()
        out.header = msg.header
        if spot is not None:
            out.point.x, out.point.y, out.point.z = spot
        else:
            out.point.z = -1.0  # 未检出标记
        self.spot_pub.publish(out)

        if self.get_parameter("publish_debug_image").value and self.debug_pub.get_subscription_count() > 0:
            dbg = frame.copy()
            cv2.drawContours(dbg, [c for c, _, _, _ in self._cand], -1, (255, 255, 0), 1)
            if spot is not None:
                cv2.circle(dbg, (int(spot[0]), int(spot[1])), 8, (0, 255, 255), 2)
            dbg_msg = cv2_to_imgmsg(dbg, "bgr8")
            dbg_msg.header = msg.header
            self.debug_pub.publish(dbg_msg)

    def _pick_spot(self, mask, val_channel):
        """返回 (u, v, strength) 或 None；strength 为峰值亮度归一化值。"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_min = self.get_parameter("area_min").value
        area_max = self.get_parameter("area_max").value
        circ_min = self.get_parameter("circularity_min").value
        self._cand = []
        best, best_peak = None, -1.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (area_min <= area <= area_max):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < circ_min:
                continue
            self._cand.append((cnt, area, circularity, perimeter))
            # 亮度加权亚像素质心 + 峰值亮度作为候选评分
            x, y, w, h = cv2.boundingRect(cnt)
            roi_mask = mask[y:y + h, x:x + w]
            roi_val = val_channel[y:y + h, x:x + w].astype(np.float32)
            roi_val[roi_mask == 0] = 0.0
            total = roi_val.sum()
            if total <= 0:
                continue
            ys, xs = np.mgrid[y:y + h, x:x + w]
            u = float((roi_val * xs).sum() / total)
            v = float((roi_val * ys).sum() / total)
            peak = float(roi_val.max())
            if peak > best_peak:
                best, best_peak = (u, v, peak / 255.0), peak
        return best


def main(args=None):
    rclpy.init(args=args)
    node = LaserSpotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
