# 单个人体检测结果，框为原图像素坐标 [左上, 右下]
float32 x1
float32 y1
float32 x2
float32 y2
float32 score
# 17 个关键点（COCO 顺序），无 pose 输出时为空数组
Keypoint[] keypoints
