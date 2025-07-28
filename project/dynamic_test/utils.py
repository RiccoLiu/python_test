# -*- coding: UTF-8 -*-
'''
Author   ： mabotan
Date     : 2025/5/28 下午3:14
Describe : 
'''
import cv2
from sklearn.linear_model import LinearRegression
import numpy as np
import math

def get_angle(xs, y_samples):
    lr = LinearRegression()
    xs, ys = xs[xs >= 0], y_samples[xs >= 0]
    if len(xs) > 1:
        lr.fit(ys[:, None], xs) # 拟合 xs = k*ys + b
        k = lr.coef_[0]
        theta = np.arctan(k) # 与y轴角度, 取值范围 [-PI/2, PI/2]
    else:
        theta = 0
    return theta

def line_accuracy(pred, gt, angle, angle_prev, skylineY, GLOBAL_CROP_A, input_size_h, R):
    if abs(angle - angle_prev) > 1.57:
        return 200

    normSkylineY = GLOBAL_CROP_A + input_size_h * 0.25 / R  # 固定值
    skylineY = normSkylineY  # 应该传入,未完成调试，暂不开通
    offsetI = len(gt) * (skylineY - normSkylineY) * R / input_size_h

    dist = 0
    num = 0
    invLen = 1.0 / (len(gt) - 1)
    dy = input_size_h * 1.0 / R / (len(gt) - 1)

    for i in range(len(gt)):
        if gt[i] < 0:
            continue
        if pred[i] < 0:
            continue
        num += 1

        if 1:  # 对于倾斜很大的边线，容易错误匹配，暂时不用该逻辑
            if i > 0 and gt[i - 1] > 0:
                x0 = gt[i - 1]
                y0 = -dy
            else:
                x0 = gt[i]
                y0 = 0

            if i + 1 < len(gt) and gt[i + 1] > 0:
                x1 = gt[i + 1]
                y1 = dy
            else:
                x1 = gt[i]
                y1 = 0

            if y1 - y0 > 1:
                ratio = (y1 - y0) / math.sqrt((y1 - y0) * (y1 - y0) + (x1 - x0) * (x1 - x0))
            else:
                ratio = abs(np.cos(angle)) # 角度权重：车道线与前方角度越小权重越大
        else:
            ratio = abs(np.cos(angle))
        # ratio = abs(ratio * ((1 - i * invLen) * 0.9 + 0.4))

        ratio1 = ((1 - (i - offsetI) * invLen) * 1.2 + 0.2) # 位置权重：车道线越远权重越大
        if ratio1 < 0.1:
            ratio1 = 0.1
        ratio = abs(ratio * ratio1)

        dist += (abs(gt[i] - pred[i]) * ratio)
    if num > 0:
        ave_dist = int(dist / num)
    else:
        ave_dist = 100
        # sum_num += np.where(np.abs(pred[i] - gt[i]) < thresh, 1., 0.)
        # k += 1
    return ave_dist  # sum_num/k

def myDraw_line(image, lane, ys, color):
    for k in range(len(lane) - 1):
        if lane[k] < 0:
            continue
        pt1 = (lane[k], int(ys[k]))
        pt2 = (lane[k + 1], int(ys[k + 1]))
        if lane[k + 1] < 0:
            continue
        cv2.line(image, pt1, pt2, color, 2)

def myDraw_circle(image, lane, ys, color, is_virtual):
    pt1s = []
    for k in range(len(lane)):
        if lane[k] < 0:
            continue
        pt1 = (lane[k], int(ys[k]))
        cv2.circle(image, pt1, 3, color, -1)
        pt1s.append(pt1)
    if is_virtual:
        cv2.putText(image, "vir", pt1s[len(pt1s) // 2], cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2)


def draw_cross(image, center, size, color=(0, 0, 255), thickness=2, circle_size=10, circle_thickness=-1):
    """
    在图像上绘制带圆心的十字标记

    参数:
    image - 目标图像
    center - 十字中心 (x, y)
    size - 十字臂长
    color - 标记颜色 (B, G, R)
    thickness - 线条厚度
    circle_size - 中心圆半径
    circle_thickness - 中心圆厚度（负值表示填充）
    """
    # 绘制水平线
    cv2.line(image,
             (center[0] - size, center[1]),
             (center[0] + size, center[1]),
             color, thickness)

    # 绘制垂直线
    cv2.line(image,
             (center[0], center[1] - size),
             (center[0], center[1] + size),
             color, thickness)

    # 绘制中心圆
    if circle_size > 0:
        cv2.circle(image, center, circle_size, color, circle_thickness)

    return image

def draw_text(image, text, point, size = cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), thickness=1):
    cv2.putText(image, text, point, size, 0.45, color, thickness)
    return image

def rmse(y_true, y_pred):
    # 转换为NumPy数组确保计算效率
    actual = np.array(y_true)
    predicted = np.array(y_pred)

    # 确保形状匹配
    if actual.shape != predicted.shape:
        raise ValueError("Shapes must match")

    return round(np.sqrt(np.mean(np.square(actual - predicted))), 2)
