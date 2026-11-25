"""YOLOv8n 通用物体检测（Ascend NPU），输出 M3 抓取感知接口。"""

import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from m1_vision import yolo_postprocess as yp
from m1_vision.acl_inference import ACL_AVAILABLE, OmModel
from m1_vision.ros_image import imgmsg_to_cv2

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__("object_detection_node")
        self.declare_parameter("om_model_path", "")
        self.declare_parameter("onnx_model_path", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("input_size", 640)
        self.declare_parameter("conf_threshold", 0.4)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter("max_det", 20)
        self.declare_parameter("device_id", 0)
        self.declare_parameter("target_fps", 5.0)
        self.declare_parameter("fps_report_frames", 100)

        self.input_size = self.get_parameter("input_size").value
        self.conf_th = self.get_parameter("conf_threshold").value
        self.nms_th = self.get_parameter("nms_threshold").value
        self.max_det = self.get_parameter("max_det").value
        self.target_fps = self.get_parameter("target_fps").value
        self._last_process = 0.0
        self.backend = self._init_backend()

        self.det_pub = self.create_publisher(
            Detection2DArray, "/m1/object_detections", 10)
        self.best_pub = self.create_publisher(Detection2D, "/detected_object", 10)
        qos = QoSProfile(depth=2)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        image_topic = self.get_parameter("image_topic").value
        self.create_subscription(Image, image_topic, self._on_image, qos)

        self._latencies = deque(maxlen=2000)
        self._report_every = self.get_parameter("fps_report_frames").value
        self._frame_count = 0
        self._window_start = time.monotonic()
        self.get_logger().info(f"object_detection_node 启动，后端：{self.backend}")

    def _init_backend(self):
        om_path = self.get_parameter("om_model_path").value
        onnx_path = self.get_parameter("onnx_model_path").value
        if om_path and ACL_AVAILABLE:
            try:
                self.model = OmModel(om_path, self.get_parameter("device_id").value)
                return "npu"
            except RuntimeError as exc:
                self.get_logger().error(f"NPU 初始化失败：{exc}，尝试 CPU 兜底")
        if onnx_path:
            try:
                import onnxruntime as ort

                self.model = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"])
                return "cpu"
            except Exception as exc:
                self.get_logger().error(f"ONNX Runtime 初始化失败：{exc}")
        self.model = None
        self.get_logger().error("无可用物体检测后端，仅发布空检测")
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
        if self.target_fps > 0 and t0 - self._last_process < 1.0 / self.target_fps:
            return
        self._last_process = t0
        frame = imgmsg_to_cv2(msg, "bgr8")
        img_lb, ratio, dwdh = yp.letterbox(frame, self.input_size)
        out = Detection2DArray()
        out.header = msg.header
        output = self._infer(yp.to_input_tensor(img_lb)) if self.model is not None else None
        if output is not None:
            boxes, scores, class_ids = yp.decode_detection_output(output, self.conf_th)
            boxes = yp.cxcywh_to_xyxy(boxes)
            keep = yp.class_aware_nms(
                boxes, scores, class_ids, self.nms_th, self.max_det)
            boxes, _ = yp.scale_back(boxes[keep], None, ratio, dwdh, frame.shape)
            for box, score, class_id in zip(boxes, scores[keep], class_ids[keep]):
                det = Detection2D()
                det.header = msg.header
                det.bbox.center.position.x = float((box[0] + box[2]) / 2.0)
                det.bbox.center.position.y = float((box[1] + box[3]) / 2.0)
                det.bbox.size_x = float(box[2] - box[0])
                det.bbox.size_y = float(box[3] - box[1])
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = COCO_NAMES[int(class_id)]
                hypothesis.hypothesis.score = float(score)
                hypothesis.pose.pose.orientation.w = 1.0
                det.results.append(hypothesis)
                out.detections.append(det)
        self.det_pub.publish(out)
        if out.detections:
            self.best_pub.publish(max(
                out.detections, key=lambda det: det.results[0].hypothesis.score))

        self._latencies.append(time.monotonic() - t0)
        self._frame_count += 1
        if self._frame_count % self._report_every == 0:
            elapsed = time.monotonic() - self._window_start
            fps = self._report_every / elapsed
            mean_ms = 1000.0 * sum(self._latencies) / len(self._latencies)
            self.get_logger().info(
                f"[{self.backend}] 物体检测 {fps:.1f} FPS，单帧均值 {mean_ms:.1f} ms")
            self._window_start = time.monotonic()

    def destroy_node(self):
        if self.backend == "npu" and self.model is not None:
            self.model.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
