"""YOLOv8n-pose 前/后处理（纯 numpy + OpenCV，NPU/CPU 推理路径共用）。"""

import cv2
import numpy as np

NUM_KEYPOINTS = 17
POSE_CHANNELS = 4 + 1 + NUM_KEYPOINTS * 3  # box + conf + 17*(x,y,c) = 56


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """等比缩放并填充到 new_shape。返回 (填充后图像, 缩放比 r, (dw, dh) 半填充量)。"""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2.0, (new_shape[0] - new_unpad[1]) / 2.0
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, (dw, dh)


def to_input_tensor(img_lb):
    """BGR uint8 -> NCHW float32 归一化张量。"""
    rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor)


def decode_pose_output(output, conf_th=0.5):
    """解析 YOLOv8n-pose 输出 (1, 56, 8400) 或 (56, 8400)。

    返回 (boxes_cxcywh[N,4], scores[N], kpts[N,17,3])，均已按 conf_th 过滤。
    """
    pred = np.squeeze(output, axis=0) if output.ndim == 3 else output
    if pred.shape[0] == POSE_CHANNELS:
        pred = pred.T
    boxes = pred[:, 0:4]
    scores = pred[:, 4]
    keep = scores > conf_th
    boxes, scores = boxes[keep], scores[keep]
    kpts = pred[keep, 5:5 + NUM_KEYPOINTS * 3].reshape(-1, NUM_KEYPOINTS, 3)
    return boxes, scores, kpts


def decode_detection_output(output, conf_th=0.5):
    """解析 YOLOv8 检测输出 (1, 84, 8400)，返回框、类别分数和类别编号。"""
    pred = np.squeeze(output, axis=0) if output.ndim == 3 else output
    if pred.shape[0] == 84:
        pred = pred.T
    boxes = pred[:, 0:4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    keep = scores > conf_th
    return boxes[keep], scores[keep], class_ids[keep]


def class_aware_nms(boxes_xyxy, scores, class_ids, iou_th=0.45, max_det=20):
    """按类别分别执行 NMS，再按置信度截取全局 max_det。"""
    kept = [nms(boxes_xyxy[class_ids == class_id], scores[class_ids == class_id],
                iou_th, max_det) for class_id in np.unique(class_ids)]
    if not kept:
        return np.empty(0, dtype=np.int64)
    class_values = np.unique(class_ids)
    indices = np.concatenate([
        np.flatnonzero(class_ids == class_id)[class_keep]
        for class_id, class_keep in zip(class_values, kept)
    ])
    return indices[np.argsort(scores[indices])[::-1][:max_det]]


def cxcywh_to_xyxy(boxes):
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def nms(boxes_xyxy, scores, iou_th=0.45, max_det=20):
    """对单类别框执行普通 NMS，返回保留索引。"""
    if len(boxes_xyxy) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) >= max_det or order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_th]
    return np.array(keep, dtype=np.int64)


def scale_back(boxes_xyxy, kpts, ratio, dwdh, orig_shape):
    """把 letterbox 坐标系的框/关键点映射回原图像素坐标，并裁剪到图像范围。"""
    dw, dh = dwdh
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    if kpts is not None and len(kpts) > 0:
        kpts = kpts.copy()
        kpts[..., 0] = ((kpts[..., 0] - dw) / ratio).clip(0, w - 1)
        kpts[..., 1] = ((kpts[..., 1] - dh) / ratio).clip(0, h - 1)
    return boxes, kpts
