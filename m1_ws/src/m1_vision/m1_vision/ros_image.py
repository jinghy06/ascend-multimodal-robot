"""sensor_msgs/Image 与 OpenCV(numpy) 图像互转。

替代 cv_bridge：板上 ROS 环境（RoboStack conda）未装 cv_bridge 且
conda 通道网络不稳，而 Image 消息本质只是字节缓冲，numpy 直转更稳。
仅支持本项目用到的编码：bgr8 / rgb8 / mono8（uint8）。
"""

import array

import cv2
import numpy as np
from sensor_msgs.msg import Image

_CHANNELS = {"bgr8": 3, "rgb8": 3, "mono8": 1}


def imgmsg_to_cv2(msg: Image, desired_encoding: str = "bgr8") -> np.ndarray:
    enc = msg.encoding.lower()
    if enc not in _CHANNELS:
        raise ValueError(f"unsupported encoding: {msg.encoding}")
    ch = _CHANNELS[enc]
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.step != msg.width * ch:  # 行对齐填充
        arr = arr.reshape(msg.height, msg.step)[:, : msg.width * ch]
    img = arr.reshape(msg.height, msg.width, ch) if ch > 1 else arr.reshape(msg.height, msg.width)
    if enc == desired_encoding:
        return img
    if enc == "rgb8" and desired_encoding == "bgr8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "mono8" and desired_encoding == "bgr8":
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc == "bgr8" and desired_encoding == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise ValueError(f"cannot convert {enc} -> {desired_encoding}")


def cv2_to_imgmsg(img: np.ndarray, encoding: str = "bgr8") -> Image:
    if encoding not in _CHANNELS:
        raise ValueError(f"unsupported encoding: {encoding}")
    ch = _CHANNELS[encoding]
    if ch == 1 and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.ascontiguousarray(img)
    msg = Image()
    msg.height, msg.width = img.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * ch
    msg.data = array.array("B", img.tobytes())
    return msg
