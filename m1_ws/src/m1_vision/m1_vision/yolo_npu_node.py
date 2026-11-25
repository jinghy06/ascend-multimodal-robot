"""yolo_npu_node：YOLOv8n-pose 人体检测推理节点（昇腾 NPU）。

推理后端优先级：OM/pyACL（NPU）→ ONNX Runtime（CPU 兜底）→ 空检测（仅保活）。
流水线：订阅图像 → letterbox 前处理（CPU）→ 推理（NPU）→ 解码 + NMS（CPU）
→ 发布 /m1/person_detections（m1_msgs/PersonDetectionArray，含 17 关键点）。

帧率统计：节点内统计端到端 FPS（订阅回调入口到发布），滚动 60 s 均值，
作为验收指标 ≥20 FPS 的实测依据（见设计文档 9.1 节）。
"""

import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from m1_msgs.msg import Keypoint, PersonDetection, PersonDetectionArray
from m1_vision import yolo_postprocess as yp
from m1_vision.acl_inference import ACL_AVAILABLE, OmModel
from m1_vision.ros_image import imgmsg_to_cv2


class YoloNpuNode(Node):
    def __init__(self):
        super().__init__("yolo_npu_node")
        self.declare_parameter("om_model_path", "")
        self.declare_parameter("onnx_model_path", "")   # CPU 兜底
        self.declare_parameter("input_size", 640)
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter("max_det", 10)
        self.declare_parameter("device_id", 0)
        self.declare_parameter("fps_report_frames", 100)

        self.input_size = self.get_parameter("input_size").value
        self.conf_th = self.get_parameter("conf_threshold").value
        self.nms_th = self.get_parameter("nms_threshold").value
        self.max_det = self.get_parameter("max_det").value

        self.backend = self._init_backend()
        self.det_pub = self.create_publisher(
            PersonDetectionArray, "/m1/person_detections", 10)

        qos = QoSProfile(depth=2)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self._on_image, qos)

        self._latencies = deque(maxlen=2000)
        self._report_every = self.get_parameter("fps_report_frames").value
        self._frame_count = 0
        self._window_start = time.monotonic()

    def _init_backend(self):
        om_path = self.get_parameter("om_model_path").value
        onnx_path = self.get_parameter("onnx_model_path").value
        if om_path and ACL_AVAILABLE:
            try:
                self.model = OmModel(om_path, self.get_parameter("device_id").value)
                self.get_logger().info(f"NPU 后端就绪：{om_path}")
                return "npu"
            except RuntimeError as exc:
                self.get_logger().error(f"NPU 初始化失败：{exc}，尝试 CPU 兜底")
        elif om_path:
            self.get_logger().error("配置了 om_model_path 但 pyACL 不可用（非 Atlas 环境？）")
        if onnx_path:
            try:
                import onnxruntime as ort

                self.model = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"])
                self.get_logger().warn(f"CPU 兜底后端：{onnx_path}（预计 8-12 FPS，仅应急）")
                return "cpu"
            except Exception as exc:  # noqa: BLE001 - 兜底失败只降级不崩溃
                self.get_logger().error(f"ONNX Runtime 初始化失败：{exc}")
        self.get_logger().error("无可用推理后端，节点将发布空检测（仅保活）")
        self.model = None
        return "none"

    def _infer(self, tensor):
        if self.backend == "npu":
            return self.model.infer(tensor)[0]
        if self.backend == "cpu":
            input_name = self.model.get_inputs()[0].name
            return self.model.run(None, {input_name: tensor})[0]
        return None

    def _on_image(self, msg: Image):
        t0 = time.monotonic()
        frame = imgmsg_to_cv2(msg, "bgr8")
        img_lb, ratio, dwdh = yp.letterbox(frame, self.input_size)
        out_msg = PersonDetectionArray()
        out_msg.header = msg.header

        output = self._infer(yp.to_input_tensor(img_lb)) if self.model is not None else None
        if output is not None:
            boxes, scores, kpts = yp.decode_pose_output(output, self.conf_th)
            keep = yp.nms(yp.cxcywh_to_xyxy(boxes), scores, self.nms_th, self.max_det)
            boxes_xyxy = yp.cxcywh_to_xyxy(boxes)[keep]
            boxes_xyxy, kpts = yp.scale_back(
                boxes_xyxy, kpts[keep], ratio, dwdh, frame.shape)
            for i in range(len(boxes_xyxy)):
                det = PersonDetection()
                det.x1, det.y1, det.x2, det.y2 = [float(v) for v in boxes_xyxy[i]]
                det.score = float(scores[keep][i])
                det.keypoints = [
                    Keypoint(x=float(k[0]), y=float(k[1]), confidence=float(k[2]))
                    for k in kpts[i]
                ]
                out_msg.detections.append(det)
        self.det_pub.publish(out_msg)

        self._latencies.append(time.monotonic() - t0)
        self._frame_count += 1
        if self._frame_count % self._report_every == 0:
            elapsed = time.monotonic() - self._window_start
            fps = self._report_every / elapsed
            mean_ms = 1000.0 * sum(self._latencies) / len(self._latencies)
            self.get_logger().info(
                f"[{self.backend}] 端到端 {fps:.1f} FPS，单帧均值 {mean_ms:.1f} ms "
                f"（最近 {len(self._latencies)} 帧）")
            self._window_start = time.monotonic()

    def destroy_node(self):
        if self.backend == "npu" and self.model is not None:
            self.model.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloNpuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
