# 相机-雷达融合后的测距目标（laser_frame 极坐标）
float32 theta      # 方位角 rad，正前方为 0，左侧为正（REP-103）
float32 distance   # 距离 m
float32 score      # 检测置信度（光斑为质心强度归一化值）
uint8 SOURCE_PERSON = 0
uint8 SOURCE_LASER_SPOT = 1
uint8 source
