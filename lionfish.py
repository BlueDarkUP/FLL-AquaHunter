import cv2
import numpy as np
import torch
import os
import sys
from time import time, sleep  # 时间库，用于计算帧率和延时
import serial
import math



# HUAWEI Ascend，向上的力量
try:
    from ais_bench.infer.interface import InferSession
except ImportError:
    print("[致命错误] 未找到 ais_bench 库。请确保已正确安装。")
    exit()

# 尝试导入工具函数
try:
    from lionfishdet_utils import letterbox, scale_coords
except ImportError:
    print("[致命错误] 未找到 lionfishdet_utils.py。请确保它在同一目录下。")
    exit()

# 尝试导入ROV控制API
try:
    from rov_api import RovVisionApi
except ImportError:
    print("[致命错误] 未找到 rov_api.py。请确保它在同一目录下。")
    exit()

# 尝试导入匈牙利算法，用于匹配
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print("[致命错误] 未安装 Scipy 库。请运行 'pip install scipy'。")
    exit()

ENABLE_GUI = False   # 是否启用图形用户界面(GUI)。False表示在终端上运行，True用于调试。

model_path = "./lionfish.om"
label_path = './class_names.txt'
SERIAL_PORT = '/dev/ttyUSB0'   # 我们用了一根USB-B在Mega Pi和Opi上通讯

BAUD_RATE = 115200 #需要和Megapi接收端的波特率一样，否则收到乱码


# 期望的目标在画面中所占的面积比例，用于判断距离是否合适
DESIRED_AREA_RATIO = 0.03
# 面积比例的容忍度，允许的误差范围
AREA_TOLERANCE = 0.04
# 抓取的目标点在画面中的归一化坐标 (0.5, 0.6) 表示比中心点稍偏下
GRAB_TARGET_POINT = (0.5, 0.6)
# 抓取对准的容忍度，用于判断是否对准
GRAB_TOLERANCE = 0.05

# 定义哪些电机的输出需要反向。True表示反向。
ROBOT_INVERTS = {
    'pitch_mid': False, 'vert_left': True, 'vert_right': False,
    'yaw_left': True, 'yaw_right': False, 'arm_extend': True,
    'arm_pitch': False, 'gripper': False
}
# 定义各个运动方向的灵敏度
ROBOT_GAINS = {
    'FORWARD_GAIN': 1.0, 'HEAVE_GAIN': 0.9, 'YAW_GAIN': 1.2,
    'BODY_PITCH_GAIN': 1.0, 'ROLL_GAIN': 1.0
}
# 定义Python脚本中的键名与发送到串口的短键名之间的映射
KEY_MAPPING = {
    'pitch_mid': 'pm', 'vert_left': 'vl', 'vert_right': 'vr',
    'yaw_left': 'yl', 'yaw_right': 'yr', 'arm_extend': 'ae',
    'arm_pitch': 'ap', 'gripper': 'gr',
    'grab_action': 'ga'  # 抓取动作的特殊指令
}


# 函数：将包含电机输出的字典格式化为串口发送的字符串
# 格式示例: "<pm:128,vl:255,yr:0,ga:1>"
def format_command_for_serial(output_dict):
    mapped_commands = []
    # 遍历映射字典，生成每个键值对
    for py_key, short_key in KEY_MAPPING.items():
        if py_key in output_dict:
            mapped_commands.append(f"{short_key}:{int(output_dict[py_key])}")
    # 打包命令
    return f"<{','.join(mapped_commands)}>"


# --- 卡尔曼滤波器跟踪器 ---
# 用于平滑目标的位置预测，即使在某几帧中检测失败也能估计其位置
class KalmanFilterTracker:
    count = 0  # 静态变量，用于为每个新跟踪器分配唯一的ID

    def __init__(self, bbox):
        # 初始化卡尔曼滤波器
        # 8: 状态变量维度 [cx, cy, w, h, d_cx, d_cy, d_w, d_h] (中心点，宽高，及它们的速度)
        # 4: 测量变量维度 [cx, cy, w, h] (我们只能直接测量位置和尺寸)
        self.kf = cv2.KalmanFilter(8, 4, 0)
        # 状态转移矩阵A: 定义了当前状态如何演变成下一状态
        self.kf.transitionMatrix = np.array(
            [[1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
            np.float32)
        # 测量矩阵H: 定义了如何从状态变量映射到测量变量
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
            np.float32)
        # 过程噪声协方差Q: 模型预测的不确定性
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        # 测量噪声协方差R: 传感器测量的不确定性
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.2
        # 将初始检测框(x1,y1,x2,y2)转换为(cx,cy,w,h)并设置为滤波器初始状态
        cx, cy, w, h = self.bbox_to_cwh(bbox)
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        # 分配ID
        self.id = KalmanFilterTracker.count
        KalmanFilterTracker.count += 1
        # 跟踪器状态变量
        self.time_since_update = 0  # 距离上次成功更新的帧数
        self.hits = 1  # 总共成功更新的次数
        self.hit_streak = 1  # 连续成功更新的次数
        self.age = 0  # 跟踪器存在的总帧数

    # 辅助函数：将 [x1, y1, x2, y2] 格式的边界框转换为 [center_x, center_y, width, height]
    def bbox_to_cwh(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return bbox[0] + w / 2, bbox[1] + h / 2, w, h

    # 预测阶段：根据上一状态预测当前状态
    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0: self.hit_streak = 0  # 如果上一帧没有更新，连续命中中断
        self.time_since_update += 1
        pred_state = self.kf.statePost
        cx, cy, w, h = pred_state[0], pred_state[1], pred_state[2], pred_state[3]
        x1, y1 = cx - w / 2, cy - h / 2
        return np.array([x1, y1, x1 + w, y1 + h])  # 返回预测的边界框 [x1, y1, x2, y2]

    # 更新阶段：使用新的测量值（检测框）来校正预测
    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        measurement = np.array(self.bbox_to_cwh(bbox), dtype=np.float32)
        self.kf.correct(measurement)

    # 获取当前最可靠的状态（校正后的状态）
    def get_state(self):
        state = self.kf.statePost
        cx, cy, w, h = state[0], state[1], state[2], state[3]
        x1, y1 = cx - w / 2, cy - h / 2
        return np.array([x1, y1, x1 + w, y1 + h])


# --- SORT算法 ---
# 计算两个边界框的IoU
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (
                bb_gt[3] - bb_gt[1]) - wh)
    return o


# SORT跟踪器主类，管理多个KalmanFilterTracker实例
class Sort:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age  # 一个跟踪器在没有匹配到检测时可以存活的最大帧数
        self.min_hits = min_hits  # 一个跟踪器需要连续匹配多少次才被认为是可靠的
        self.iou_threshold = iou_threshold  # IoU阈值，用于判断检测和跟踪是否匹配
        self.trackers = []  # 存储所有活跃的KalmanFilterTracker实例
        self.frame_count = 0  # 帧计数器

    # 每帧调用此方法来更新跟踪状态
    def update(self, dets):  # dets是当前帧的所有检测结果
        self.frame_count += 1
        # 1. 预测所有现有跟踪器的位置
        trks = np.zeros((len(self.trackers), 4))
        to_del = []  # 记录需要删除的跟踪器
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()  # 调用卡尔曼滤波器的预测
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)): to_del.append(t)  # 如果预测结果无效，则标记删除
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del): self.trackers.pop(t)

        # 2. 计算当前帧检测(dets)与预测的跟踪(trks)之间的IoU矩阵
        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks): iou_matrix[d, t] = iou(det, trk)

        # 3. 使用匈牙利算法（或线性指派问题求解器）找到最佳匹配
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # 加负号因为算法求最小值，而我们需要最大化IoU
        matched_indices = np.array(list(zip(row_ind, col_ind)))

        # 4. 分离出未匹配的检测和未匹配的跟踪器
        unmatched_detections = []
        if matched_indices.shape[0] > 0:
            for d, det in enumerate(dets):
                if d not in matched_indices[:, 0]: unmatched_detections.append(d)
        else:
            unmatched_detections = list(range(len(dets)))
        unmatched_trackers = []
        if matched_indices.shape[0] > 0:
            for t, trk in enumerate(trks):
                if t not in matched_indices[:, 1]: unmatched_trackers.append(t)
        else:
            unmatched_trackers = list(range(len(trks)))

        # 5. 过滤掉IoU低于阈值的匹配
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) > 0:
            matches = np.concatenate(matches, axis=0)
        else:
            matches = np.empty((0, 2), dtype=int)

        # 6. 更新状态
        #   - 对于匹配上的，用检测框更新对应的卡尔曼滤波器
        for m in matches: self.trackers[m[1]].update(dets[m[0], :])
        #   - 对于未匹配的检测，创建新的跟踪器
        for i in unmatched_detections: self.trackers.append(KalmanFilterTracker(dets[i, :]))

        # 7. 清理并返回结果
        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # 只有当跟踪器是可靠的（被更新过且达到最小命中数），才返回其位置和ID
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))  # [x1,y1,x2,y2,id]
            i -= 1
            #   - 对于长时间未更新的跟踪器（未匹配的跟踪器），将其删除
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0: return np.concatenate(ret)
        return np.empty((0, 5))


# --- 模型推理相关函数 ---
# 图像预处理
def preprocess_image(image, input_shape, bgr2rgb=True):
    # letterbox函数将图像缩放并填充以适应模型输入尺寸，同时保持纵横比
    img, scale_ratio, pad_size = letterbox(image, new_shape=input_shape)
    if bgr2rgb: img = img[:, :, ::-1]  # BGR to RGB
    # HWC to CHW, 并确保内存连续
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img, scale_ratio, pad_size


# 从txt文件加载类别标签
def get_labels_from_txt(path):
    labels_dict = dict()
    with open(path) as f:
        for cat_id, label in enumerate(f.readlines()): labels_dict[cat_id] = label.strip()
    return labels_dict


# YOLOv8模型输出的后处理
def postprocess_yolov8(prediction, conf_thres=0.25, iou_thres=0.45):
    try:
        from torchvision.ops import nms  # 尝试使用torchvision的高效NMS
    except ImportError:
        print("[警告] 未找到 torchvision.ops.nms。NMS 可能会变慢。"); pass
    # 转换数据格式并筛选掉置信度低的候选框
    prediction = torch.from_numpy(prediction).squeeze(0).t()
    boxes, scores = prediction[:, :4], prediction[:, 4:]
    max_scores, class_ids = scores.max(1)
    candidates = max_scores > conf_thres
    boxes, max_scores, class_ids = boxes[candidates], max_scores[candidates], class_ids[candidates]
    if boxes.shape[0] == 0: return np.array([])
    # 将 [cx, cy, w, h] 格式的框转换为 [x1, y1, x2, y2]
    xyxy_boxes = torch.zeros_like(boxes)
    xyxy_boxes[:, 0], xyxy_boxes[:, 1] = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2;
    xyxy_boxes[:, 2], xyxy_boxes[:, 3] = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
    # 执行非极大值抑制 (NMS)
    indices = nms(xyxy_boxes, max_scores, iou_thres)
    final_detections = []
    for i in indices:
        final_detections.append([
            xyxy_boxes[i, 0].item(), xyxy_boxes[i, 1].item(),
            xyxy_boxes[i, 2].item(), xyxy_boxes[i, 3].item(),
            max_scores[i].item(), class_ids[i].item()
        ])  # [x1,y1,x2,y2,conf,class_id]
    return np.array(final_detections)


# --- 主逻辑与控制 ---
# 在控制台打印详细的调试信息
def print_detailed_debug_info(frame_num, debug_data):
    os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
    print(f"{'=' * 20} 帧 {frame_num} {'=' * 20}")
    print(f"可抓取状态: {debug_data.get('grabbable_status', '未知')}")
    final_outputs = debug_data.get('final_outputs', {})
    if final_outputs:
        output_str = ""
        for key, _ in KEY_MAPPING.items():
            output_str += f"{key}: {int(final_outputs.get(key, 'N/A')):<5} | "
        print(output_str.strip().rstrip('|'))
    sys.stdout.flush()  # 立即刷新输出缓冲区


# 核心函数：集检测、跟踪和控制于一体
def detect_track_and_control(model, mot_tracker, frame, labels, cfg, rov_controller, mega_pi_serial, enable_gui=False):
    debug_data = {}  # 用于存储调试信息
    is_grabbable = False  # 当前是否满足抓取条件

    # 1. 预处理图像
    img, scale_ratio, pad_size = preprocess_image(frame, cfg['input_shape'])
    img = img / 255.0  # 归一化
    # 2. 模型推理
    raw_outputs = model.infer([img])
    # 3. 后处理推理结果
    detections = postprocess_yolov8(raw_outputs[0], conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    # 4. 将检测框坐标从模型输入尺寸映射回原始图像尺寸
    if len(detections) > 0:
        scale_coords(cfg['input_shape'], detections[:, :4], frame.shape, ratio_pad=(scale_ratio, pad_size))
    # 5. 更新SORT跟踪器
    tracked_objects = mot_tracker.update(detections[:, :5] if len(detections) > 0 else np.empty((0, 5)))
    # 6. 选择主要目标（这里简单地选择第一个跟踪到的目标）
    main_target = tracked_objects[0] if len(tracked_objects) > 0 else None

    # 初始化控制指令字典
    commands = {'forward': 0.0, 'heave': 0.0, 'yaw': 0.0, 'body_pitch': 0.0, 'roll': 0.0, 'arm_extend': 0.0,
                'arm_pitch': 0.0, 'gripper': 0.0, 'grab_action': 0}

    # 7. 如果有主要目标，则执行闭环控制
    if main_target is not None:
        (h, w) = frame.shape[:2]
        # 计算抓取目标点在图像上的像素坐标
        grab_target_px = (int(GRAB_TARGET_POINT[0] * w), int(GRAB_TARGET_POINT[1] * h))
        x1, y1, x2, y2, _ = main_target
        # 计算目标的中心点
        target_center_px = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        # 计算位置误差
        error_x = target_center_px[0] - grab_target_px[0]
        error_y = target_center_px[1] - grab_target_px[1]

        # 判断是否对准
        distance_to_grab_point = math.sqrt(error_x ** 2 + error_y ** 2)
        is_aligned_for_grab = distance_to_grab_point < (w * GRAB_TOLERANCE)
        # 判断距离是否合适（通过面积）
        target_area = (x2 - x1) * (y2 - y1)
        current_area_ratio = target_area / (w * h)
        is_at_correct_distance = abs(current_area_ratio - DESIRED_AREA_RATIO) < AREA_TOLERANCE

        # 如果对准且距离合适，则满足抓取条件
        if is_aligned_for_grab and is_at_correct_distance:
            is_grabbable = True
            debug_data['grabbable_status'] = "是"
        else:
            # 否则，根据误差进行调整
            debug_data['grabbable_status'] = "否 (正在对准...)"
            # P控制器：根据面积比例调整前进/后退
            if not is_at_correct_distance:
                commands['forward'] = 0.6 if current_area_ratio < DESIRED_AREA_RATIO else -0.6
            # P控制器：根据x, y误差调整偏航和升降
            commands['yaw'] = np.clip(error_x / (w * 0.5), -1.0, 1.0)
            commands['heave'] = np.clip(-error_y / (h * 0.5), -1.0, 1.0)
            commands['arm_pitch'] = np.clip(-error_y / (h * 0.5), -1.0, 1.0)  # 机械臂俯仰也联动
    else:
        # 如果没有目标，则慢速旋转搜索
        debug_data['grabbable_status'] = "否 (正在搜索...)"
        commands['yaw'] = 0.2

    # 设置抓取动作标志位
    commands['grab_action'] = 1 if is_grabbable else 0

    # 8. 将高级指令通过ROV API转换为底层电机输出
    final_outputs = rov_controller.get_outputs(tuple(commands.values()))

    # 手动覆盖一些值，确保这些动作在自动模式下被禁用或由grab_action控制
    final_outputs['gripper'] = 0
    final_outputs['arm_extend'] = 0
    final_outputs['grab_action'] = commands['grab_action']

    debug_data['final_outputs'] = final_outputs
    print_detailed_debug_info(mot_tracker.frame_count, debug_data)

    # 9. 如果串口已连接，发送指令
    if mega_pi_serial and mega_pi_serial.is_open:
        command_string = format_command_for_serial(final_outputs)
        mega_pi_serial.write(command_string.encode('utf-8'))

    # 10. 如果启用GUI，绘制调试信息到画面上
    if enable_gui:
        (h, w) = frame.shape[:2]
        grab_target_px = (int(GRAB_TARGET_POINT[0] * w), int(GRAB_TARGET_POINT[1] * h))
        np.random.seed(42)  # 固定随机种子，确保颜色一致
        colors = np.random.randint(0, 255, size=(KalmanFilterTracker.count + 1, 3), dtype=np.uint8)
        # 绘制抓取目标点
        cv2.circle(frame, grab_target_px, 10, (0, 0, 255), 2)
        cv2.line(frame, (grab_target_px[0] - 15, grab_target_px[1]), (grab_target_px[0] + 15, grab_target_px[1]),
                 (0, 0, 255), 1)
        cv2.line(frame, (grab_target_px[0], grab_target_px[1] - 15), (grab_target_px[0], grab_target_px[1] + 15),
                 (0, 0, 255), 1)
        # 绘制所有跟踪到的对象
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            color = colors[int(obj_id) % (KalmanFilterTracker.count + 1)].tolist()
            is_main_target = main_target is not None and obj_id == main_target[4]
            thickness = 4 if is_main_target else 2  # 主目标框加粗
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if is_main_target:
                # 绘制主目标的中心点
                target_center_px = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(frame, target_center_px, 5, (255, 0, 0), -1)

    return frame


# --- 程序入口 ---
if __name__ == "__main__":
    # 1. 初始化ROV控制器API
    rov_controller = RovVisionApi(gain_settings=ROBOT_GAINS, invert_settings=ROBOT_INVERTS)

    # 2. 初始化串口
    mega_pi_serial = None
    try:
        mega_pi_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"串口 {SERIAL_PORT} 打开成功。等待设备初始化...")
        sleep(2)  # 等待下位机重启
    except serial.SerialException as e:
        print(f"[警告] 无法打开串口 {SERIAL_PORT}。将以模拟模式运行。错误: {e}")

    # 3. 加载模型和配置
    print("正在加载模型...")
    model = InferSession(0, model_path)  # 0表示使用NPU设备ID 0
    labels = get_labels_from_txt(label_path)
    config = {'conf_thres': 0.55, 'iou_thres': 0.4, 'input_shape': [320, 320]}
    mot_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # 4. 打开摄像头
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头
    if not cap.isOpened():
        print("[致命错误] 无法打开摄像头。")
        exit()

    prev_time = 0
    try:
        # 5. 主循环
        while True:
            ret, frame = cap.read()  # 读取一帧
            if not ret:
                sleep(1)  # 如果读取失败，等待一下再试
                continue

            # 调用核心处理函数
            processed_frame = detect_track_and_control(
                model, mot_tracker, frame, labels, config, rov_controller, mega_pi_serial,
                enable_gui=ENABLE_GUI
            )

            # 如果启用GUI，显示画面和FPS
            if ENABLE_GUI:
                curr_time = time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('狮子鱼自动抓取系统', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
                    break
            else:
                # 在无GUI模式下，短暂休眠以避免CPU占用100%
                sleep(0.01)

    except KeyboardInterrupt:
        print("\n程序被用户中断 (Ctrl+C)。")
    finally:
        # 6. 清理资源
        print("正在清理资源...")
        cap.release()
        if mega_pi_serial and mega_pi_serial.is_open:
            # 发送停止指令，让机器人停下来
            stop_outputs = rov_controller.get_outputs((0, 0, 0, 0, 0, 0, 0, 0, 0))
            stop_outputs['gripper'] = 0
            stop_outputs['arm_extend'] = 0
            stop_outputs['grab_action'] = 0
            stop_string = format_command_for_serial(stop_outputs)
            mega_pi_serial.write(stop_string.encode('utf-8'))
            sleep(0.1)
            mega_pi_serial.close()
        if ENABLE_GUI:
            cv2.destroyAllWindows()
        print("程序已结束。")