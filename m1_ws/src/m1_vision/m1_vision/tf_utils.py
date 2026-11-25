"""TF/几何小工具：不依赖 tf_transformations，避免额外的包依赖。"""

import math

import numpy as np


def quat_rotate(q, v):
    """四元数 (x, y, z, w) 旋转向量 v=(x, y, z)。返回 numpy (3,)。"""
    x, y, z, w = q
    # t = 2 * q_vec × v
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    # v' = v + w*t + q_vec × t
    rx = v[0] + w * tx + (y * tz - z * ty)
    ry = v[1] + w * ty + (z * tx - x * tz)
    rz = v[2] + w * tz + (x * ty - y * tx)
    return np.array([rx, ry, rz])


def wrap_angle(a):
    """角度归一化到 (-pi, pi]。"""
    return math.atan2(math.sin(a), math.cos(a))
