# 导入所需库
import time
import cv2
import numpy as np
import torch
import torchvision


# letterbox 函数：调整图像大小并进行填充以适应模型输入
# 功能：将任意尺寸的图像，在保持其原始宽高比的前提下，缩放并填充成一个固定尺寸的正方形（如320x320）
# 返回：处理后的图像，缩放比例，填充尺寸
def letterbox(img, new_shape=(320, 320), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # 获取原始图像的形状 [高, 宽]
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例 r
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 如果不允许放大，则r最大为1.0
        r = min(r, 1.0)

    # 计算缩放后的尺寸和所需的填充量
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽和高的填充量

    dw /= 2  # 将填充平均分配到两侧
    dh /= 2

    # 如果尺寸发生变化，则进行缩放
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 计算上下左右的填充值
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 添加边框（填充）
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


# 边界框格式转换：从 [x1, y1, x2, y2] 转换为 [中心x, 中心y, 宽, 高]
def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# 非极大值抑制 (Non-Maximum Suppression, NMS)
# 功能：在众多重叠的检测框中，筛选出置信度最高且不与其它已选框过度重叠的框。
def non_max_suppression(
        prediction,
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # IoU阈值
        classes=None,  # 是否只保留特定类别
        agnostic=False,  # 是否进行跨类别的NMS
        multi_label=False,  # 是否一个框可以有多个标签
        labels=(),
        max_det=300,  # 每张图最大检测数量
        nm=0,
):
    """对推理结果执行NMS，以抑制重叠的检测。"""

    # ... [此处为YOLOv5官方NMS代码的精简版，逻辑较为复杂，核心步骤如下] ...
    # 1. 检查输入并设置参数
    # 2. 筛选：只保留置信度高于 `conf_thres` 的候选框
    # 3. 格式转换：将 (center_x, center_y, width, height) 转换为 (x1, y1, x2, y2)
    # 4. 核心NMS：对每个类别，使用 torchvision.ops.nms 计算，消除IoU大于 `iou_thres` 的重叠框
    # 5. 返回最终筛选后的检测列表，每个检测为 [xyxy, conf, cls]

    bs = prediction.shape[0]  # 批处理大小
    nc = prediction.shape[2] - nm - 5  # 类别数量
    xc = prediction[..., 4] > conf_thres  # 候选框

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        else:
            x = x[x[:, 4].argsort(descending=True)]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]

    return output


# 边界框格式转换：从 [中心x, 中心y, 宽, 高] 转换为 [x1, y1, x2, y2]
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# 坐标缩放函数
# 功能：将检测框的坐标从模型输入尺寸（如320x320）转换回原始图像的尺寸
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]  # 缩放比例
        pad = ratio_pad[1]  # 填充量

    # 减去填充
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    # 除以缩放比例
    coords[:, :4] /= gain
    # 裁剪坐标，确保不超出原始图像边界
    clip_coords(coords, img0_shape)

    return coords


# 裁剪坐标函数
# 功能：确保边界框的坐标值不会超出图像的实际高和宽
def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


# NMS 的一个包装函数，便于调用
def nms(box_out, conf_thres=0.4, iou_thres=0.5):
    try:
        # 尝试使用支持多标签的NMS
        boxout = non_max_suppression(box_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
    except:
        # 如果失败，使用标准NMS
        boxout = non_max_suppression(box_out, conf_thres=conf_thres, iou_thres=iou_thres)
    return boxout