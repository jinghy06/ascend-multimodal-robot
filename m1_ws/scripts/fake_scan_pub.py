#!/usr/bin/env python3
"""发布恒定 /scan（联调工具）：默认全场 2.0 m，可指定一处近距簇。

用法：python3 fake_scan_pub.py [簇中心角度rad] [簇距离m] [簇半宽rad]
"""
import math
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

THETA = float(sys.argv[1]) if len(sys.argv) > 1 else None
DIST = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
HALF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05


class FakeScan(Node):
    def __init__(self):
        super().__init__("fake_scan_pub")
        self.pub = self.create_publisher(LaserScan, "/scan", 10)
        self.create_timer(0.1, self.tick)

    def tick(self):
        n = 360
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = "laser_frame"
        scan.angle_min, scan.angle_max = -math.pi, math.pi
        scan.angle_increment = 2 * math.pi / n
        scan.range_min, scan.range_max = 0.12, 8.0
        ranges = [2.0] * n
        if THETA is not None:
            for i in range(n):
                a = scan.angle_min + i * scan.angle_increment
                if abs(math.atan2(math.sin(a - THETA), math.cos(a - THETA))) < HALF:
                    ranges[i] = DIST
        scan.ranges = ranges
        self.pub.publish(scan)


rclpy.init()
node = FakeScan()
rclpy.spin(node)
