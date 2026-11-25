"""target_tracking_node：目标跟踪与丢失恢复 + 激光笔导航目标解算。

跟随部分（设计文档 5.5 节）：
  恒定速度模型卡尔曼滤波，状态 (θ, r, ω, v_r)；最近邻 + 马氏距离门限关联；
  TRACKING / LOST / RECOVERING 状态机，20 Hz 向 M4 发布 /m1/follow_target
  （base_link 直角坐标），LOST 时保持最后一次预测并置状态 LOST。

激光笔部分（设计文档 5.4 节）：
  /m1/laser_spot_ranged 经 TF 转到 map 系，连续稳定（窗口内散布 ≤ 阈值）
  后取均值，向机器人方向内缩 safety_offset 作为导航目标，发布 /m1/laser_goal。
"""

import math
import time
from collections import deque

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  注册 PointStamped/PoseStamped 的 TF 变换
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from m1_msgs.msg import RangedTarget, RangedTargetArray
from m1_vision.tf_utils import wrap_angle

# 状态机
IDLE, TRACKING, LOST, RECOVERING = "IDLE", "TRACKING", "LOST", "RECOVERING"


class PolarKalman:
    """极坐标 (θ, r) 匀速模型卡尔曼滤波：x = [θ, r, ω, v_r]。"""

    def __init__(self, q_theta=0.05, q_r=0.05, r_theta=0.02, r_r=0.05):
        self.x = None
        self.P = None
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        self.R = np.diag([r_theta, r_r])
        self.q_theta, self.q_r = q_theta, q_r

    def _F(self, dt):
        return np.array([[1.0, 0.0, dt, 0.0],
                         [0.0, 1.0, 0.0, dt],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    def init(self, theta, r):
        self.x = np.array([theta, r, 0.0, 0.0])
        self.P = np.diag([0.1, 0.1, 1.0, 1.0])

    def predict(self, dt):
        if self.x is None:
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.x[0] = wrap_angle(self.x[0])
        Q = np.diag([self.q_theta * dt, self.q_r * dt, self.q_theta, self.q_r])
        self.P = F @ self.P @ F.T + Q

    def gate_distance(self, theta, r):
        """观测与预测的马氏距离（2 自由度，chi2=9.21 对应 99%）。"""
        z = np.array([theta, r])
        z_pred = self.H @ self.x
        y = z - z_pred
        y[0] = wrap_angle(y[0])
        S = self.H @ self.P @ self.H.T + self.R
        return float(y @ np.linalg.inv(S) @ y)

    def update(self, theta, r):
        z = np.array([theta, r])
        z_pred = self.H @ self.x
        y = z - z_pred
        y[0] = wrap_angle(y[0])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[0] = wrap_angle(self.x[0])
        self.P = (np.eye(4) - K @ self.H) @ self.P


class TargetTrackingNode(Node):
    def __init__(self):
        super().__init__("target_tracking_node")
        # 跟随跟踪参数
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("lock_hits", 3)
        self.declare_parameter("lost_misses", 10)
        self.declare_parameter("gate_chi2", 9.21)
        self.declare_parameter("lost_reset_timeout", 5.0)  # LOST 超时后清空轨迹，视为新目标
        self.declare_parameter("laser_frame", "laser_frame")
        self.declare_parameter("base_frame", "base_link")
        # 激光笔目标参数
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("spot_stable_window", 2.0)    # s
        self.declare_parameter("spot_stable_radius", 0.15)   # m，窗口内散布上限
        self.declare_parameter("spot_min_points", 5)
        self.declare_parameter("goal_safety_offset", 0.3)    # m，向机器人内缩
        self.declare_parameter("goal_cooldown", 3.0)         # s，两次发目标最小间隔

        self.laser_frame = self.get_parameter("laser_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.map_frame = self.get_parameter("map_frame").value
        self.lock_hits = self.get_parameter("lock_hits").value
        self.lost_misses = self.get_parameter("lost_misses").value
        self.gate_chi2 = self.get_parameter("gate_chi2").value

        self.kf = PolarKalman()
        self.state = IDLE
        self.hits = 0
        self.misses = 0
        self._lost_since = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.follow_pub = self.create_publisher(PointStamped, "/m1/follow_target", 10)
        self.status_pub = self.create_publisher(String, "/m1/target_status", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/m1/laser_goal", 10)

        self.create_subscription(RangedTargetArray, "/m1/ranged_targets",
                                 self._on_targets, 10)
        self.create_subscription(PointStamped, "/m1/laser_spot_ranged",
                                 self._on_spot_ranged, 10)

        self._spot_window = deque()   # (t, x_map, y_map)
        self._last_goal_time = 0.0

        rate = self.get_parameter("publish_rate").value
        self._dt = 1.0 / rate
        self.create_timer(self._dt, self._tick)
        self.get_logger().info(f"target_tracking_node 启动，发布 {rate} Hz")

    # ---- 跟随跟踪 ----
    def _on_targets(self, msg: RangedTargetArray):
        persons = [t for t in msg.targets if t.source == RangedTarget.SOURCE_PERSON]
        if not persons:
            self.misses += 1
            self.hits = 0
            self._update_state(got_measurement=False)
            return
        if self.kf.x is None:
            best = max(persons, key=lambda t: t.score)
            self.kf.init(best.theta, best.distance)
            self.hits, self.misses = 1, 0
        else:
            # 最近邻 + 马氏距离门限关联
            best = min(persons, key=lambda t: self.kf.gate_distance(t.theta, t.distance))
            if self.kf.gate_distance(best.theta, best.distance) <= self.gate_chi2:
                self.kf.update(best.theta, best.distance)
                self.hits += 1
                self.misses = 0
            else:
                self.misses += 1
                self.hits = 0
        self._update_state(got_measurement=self.hits > 0)

    def _update_state(self, got_measurement):
        prev = self.state
        if self.state == IDLE:
            if self.hits >= self.lock_hits:
                self.state = TRACKING
                self.get_logger().info("目标锁定 → TRACKING")
        elif self.state == TRACKING:
            if self.misses >= self.lost_misses:
                self.state = LOST
                self._lost_since = time.monotonic()
                self.get_logger().warn("目标丢失 → LOST")
        elif self.state == LOST:
            if got_measurement:
                self.state = RECOVERING
                self.get_logger().info("重新捕获候选 → RECOVERING")
        elif self.state == RECOVERING:
            if self.hits >= self.lock_hits:
                dt = time.monotonic() - (self._lost_since or time.monotonic())
                self.state = TRACKING
                self.get_logger().info(f"目标恢复 → TRACKING（丢失恢复耗时 {dt:.2f} s，指标 ≤3 s）")
            elif self.misses >= self.lost_misses:
                self.state = LOST
                self._lost_since = time.monotonic()
        if prev != self.state:
            msg = String()
            msg.data = self.state
            self.status_pub.publish(msg)

    def _tick(self):
        """20 Hz：卡尔曼预测并发布跟随目标（LOST 保持最后预测值）。"""
        if self.state == LOST and self._lost_since is not None:
            timeout = self.get_parameter("lost_reset_timeout").value
            if time.monotonic() - self._lost_since > timeout:
                # 丢失超时：清空轨迹，下次检测按新目标重新初始化（文档 5.5）
                self.kf.x = None
                self.state = IDLE
                self.hits = self.misses = 0
                self._lost_since = None
                msg = String()
                msg.data = self.state
                self.status_pub.publish(msg)
                self.get_logger().info("丢失超时，轨迹清空 → IDLE（下次检测视为新目标）")
                return
        if self.kf.x is None or self.state == IDLE:
            return
        self.kf.predict(self._dt)
        theta, r = float(self.kf.x[0]), float(self.kf.x[1])
        # laser_frame 极坐标 → base_frame 直角坐标
        pt = PointStamped()
        pt.header.frame_id = self.laser_frame
        pt.point.x = r * math.cos(theta)
        pt.point.y = r * math.sin(theta)
        pt.point.z = 0.0
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.laser_frame, Time())
            pt = tf2_geometry_msgs.do_transform_point(pt, tf)
        except TransformException:
            pass  # TF 暂不可用时直接按 laser_frame 发布（静止安装近似重合）
        pt.header.stamp = self.get_clock().now().to_msg()
        self.follow_pub.publish(pt)

    # ---- 激光笔导航目标 ----
    def _on_spot_ranged(self, msg: PointStamped):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id, Time())
            p_map = tf2_geometry_msgs.do_transform_point(msg, tf)
        except TransformException as exc:
            self.get_logger().warn(f"map TF 不可用，忽略光斑点：{exc}",
                                   throttle_duration_sec=2.0)
            return
        now = time.monotonic()
        window = self.get_parameter("spot_stable_window").value
        self._spot_window.append((now, p_map.point.x, p_map.point.y))
        while self._spot_window and now - self._spot_window[0][0] > window:
            self._spot_window.popleft()

        min_pts = self.get_parameter("spot_min_points").value
        if len(self._spot_window) < min_pts:
            return
        pts = np.array([(p[1], p[2]) for p in self._spot_window])
        center = pts.mean(axis=0)
        spread = np.linalg.norm(pts - center, axis=1).max()
        if spread > self.get_parameter("spot_stable_radius").value:
            return
        if now - self._last_goal_time < self.get_parameter("goal_cooldown").value:
            return

        # 向机器人位置内缩 safety_offset，避免目标点贴墙不可达
        offset = self.get_parameter("goal_safety_offset").value
        try:
            tf_base = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
            robot = np.array([tf_base.transform.translation.x,
                              tf_base.transform.translation.y])
        except TransformException:
            robot = np.zeros(2)
        direction = center - robot
        norm = np.linalg.norm(direction)
        if norm > offset:
            goal_xy = center - offset * direction / norm
        else:
            goal_xy = robot

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.map_frame
        goal.pose.position.x = float(goal_xy[0])
        goal.pose.position.y = float(goal_xy[1])
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        self._last_goal_time = now
        self._spot_window.clear()
        self.get_logger().info(
            f"激光笔导航目标：map ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})，"
            f"原始光斑点 ({center[0]:.2f}, {center[1]:.2f})")


def main(args=None):
    rclpy.init(args=args)
    node = TargetTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
