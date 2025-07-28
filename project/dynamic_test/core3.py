# -*- coding: UTF-8 -*-
'''
Author   ： mabotan
Date     : 2025/5/28 上午11:46
Describe : 
'''
import cv2
import json
import os
import copy
from collections import defaultdict
from loguru import logger
from glob import glob
import numpy as np
from utils import get_angle, line_accuracy, myDraw_line, myDraw_circle, draw_cross, rmse, draw_text
from scipy import interpolate
import pandas as pd
import argparse
from ipm_transformer import IpmTransformer

logger.add("my_log.log")

class Metric():
    def __init__(self, image_path, annotation_path, pred_path, test_pre_num = 2, img_save_path = "./images"):
        """
        test_pre_num: 测试前n条车道
        """
        self.annotation_path = annotation_path
        self.pred_path = pred_path
        self.origin_img_path = image_path
        self.img_save_path = img_save_path

        # 初始化测试先验相关
        self.input_size_w = 960
        self.input_size_h = 192
        self.image_height =1080
        self.image_width = 1920
        self.R = 0.5
        self.h_samples = np.linspace(0, self.input_size_h / self.R, num=32)
        self.ys = self.h_samples
        self.crop_a = self.GLOBAL_CROP_A = int((self.image_height - self.input_size_h / self.R) * 0.75)
        self.CATE_LIST = ['Ignore', 'SingleSolid', 'SingleDotted', 'DoubleSolid', 'SolidDotted', 'DottedSolid', 'DoubleDotted', 'DotDashed', 'Roadline', 'Fence']
        self.COLOR_LIST = ['other', 'white', 'yellow', 'blue', 'orange']
        self.TEST_VIRTUAL = False
        self.test_pre_num = test_pre_num
        self.breakPointScoreYArrayRows = 8
        self.MAX_DIS = 30
        self.tp_color = [0, 255,0] # 正确，绿色
        self.fp_color = [0, 255, 255] # 误检，黄色
        self.fn_color = [0, 0, 255] # 漏检，红色
        self.center_position = 2 * len(self.ys) // 3

        self.frame_interval = 33 # ms
        self.ipm_trans_ = IpmTransformer(ipm_range=(15, 10), reso=0.02, start_forward=1.0, is_distort=True)

        logger.info("正在加载标签")
        self._get_annotation()
        logger.info("正在加载预测")
        self._get_prediction()
        logger.info("正在向预测结果中按照魔毯规则添加车道线")
        self._add_motan()

        self.fn_self = 0
        self.fp_self = 0
        self.contain_virtual = 0
        self.total_virtuals = 0
        self.total_gt_lanes = 0
        self.total_imgs = 0

        # 临时调试
        self.has_virtual = []

    def _get_annotation(self):
        self.annotation = defaultdict(dict)
        with open(self.annotation_path, 'r') as f:
            annotations = json.load(f)
            for item in annotations["items"]:
                id = item["id"]
                frame_id = int(id.split("_")[-1])
                video_name = id.replace("_" + str(frame_id), "")
                self.annotation[video_name][frame_id] = item
            self.category = [i["name"] for i in annotations["categories"]["label"]["labels"]]

    def _get_prediction(self):
        videos = os.listdir(self.pred_path)
        self.prediction = defaultdict(dict)
        for video in videos:
            pred_results_path = os.path.join(self.pred_path, video)
            results = sorted(glob(pred_results_path + "/*.txt"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for idx, result_path in enumerate(results):
                frame_id = int(os.path.basename(result_path).split("_")[-1].split(".")[0])
                video_name = os.path.basename(result_path).replace("_" + str(frame_id) + ".txt", "")
                pred_result = self.pred_parser(result_path)
                self.prediction[video_name][frame_id] = pred_result

    def _lane_width_alignment(self, lanes, prev_lanes):
        lane_width = [j - i if i > 0 and j > 0 else -2 for i, j in zip(lanes["pred_lanes"][0], lanes["pred_lanes"][1])]
        prev_lane_width = [j - i if i > 0 and j > 0 else -2 for i, j in zip(prev_lanes["pred_lanes"][0], prev_lanes["pred_lanes"][1])]
        
        assert len(lane_width) == len(prev_lane_width)
        
        offset = None
        for i in range(len(lane_width)):
            if lane_width[i] > 0 and prev_lane_width[i] > 0:
                offset = prev_lane_width[i] - lane_width[i]
                break
        
        if offset is not None:
            lanes["pred_lanes"][0] = [ x - offset // 2 if x != -2 else -2 for x in lanes["pred_lanes"][0]]
            lanes["pred_lanes"][1] = [ x + offset // 2 if x != -2 else -2 for x in lanes["pred_lanes"][1]]
        return lanes
    def _add_motan(self):
        # self.prediction_add_motan = self.prediction.copy()
        self.prediction_add_motan = copy.deepcopy(self.prediction)
        all_video_nams = self.prediction_add_motan.keys()
        for video_name in all_video_nams:
            # last_key = None
            # for frame_id in self.prediction_add_motan[video_name].keys():
            #     if last_key is not None and len(self.prediction_add_motan[video_name][frame_id]["pred_lanes"]) == 0 and len(self.prediction_add_motan[video_name][last_key]["pred_lanes"]) != 0:
            #         self.prediction_add_motan[video_name][frame_id] = self.prediction_add_motan[video_name][last_key]
            #     last_key = frame_id

            for frame_id in self.prediction_add_motan[video_name].keys():                
                if len(self.prediction_add_motan[video_name][frame_id]["pred_lanes"]) != 0:
                    continue

                prev_frame_id = -1
                for id in range(frame_id - 1, 0, -1):
                    if len(self.prediction[video_name][id]["pred_lanes"]) >= 2:
                        prev_frame_id = id
                        break    
                    
                freeze_lanes = {}
                if prev_frame_id <= 0:
                    freeze_lanes = self.add_freeze_lanes()    
                else:
                    lost_time = (frame_id - prev_frame_id) * self.frame_interval
                    if lost_time < 1000.0: # ms
                        freeze_lanes = copy.deepcopy(self.prediction[video_name][prev_frame_id])
                    else:
                        freeze_lanes = self.add_freeze_lanes()    
                        freeze_lanes = self._lane_width_alignment(freeze_lanes, self.prediction[video_name][prev_frame_id].copy())
                self.prediction_add_motan[video_name][frame_id].update(freeze_lanes) 

    def _whole_lane_eval(self, lane_annotation, lane_pred):
        tp, fp, fn, tn = 0, 0, 0,0
        gt_lane = len(lane_annotation["gt_lanes"])
        pred_lane = len(lane_pred["pred_lanes"])
        gt_lane_cnt = gt_lane // 2
        pred_lane_cnt = pred_lane // 2
        if gt_lane_cnt == pred_lane_cnt and gt_lane_cnt != 0:
            tp = 1
        elif gt_lane_cnt == pred_lane_cnt and gt_lane_cnt == 0:
            tn = 1
        elif gt_lane_cnt == 0 and pred_lane_cnt > 0:
            fp = 1
        elif gt_lane_cnt > 0 and pred_lane_cnt == 0:
            fn = 1
        else:
            raise
        return tp, fp, fn, tn

    def lane_line_distance_eval(self, lane_annotation, lane_pred):
        gt_lane = lane_annotation["gt_lanes"]
        pred_lane = lane_pred["pred_lanes"]
        line_0, line_1 = "null", "null"
        if len(gt_lane) > 0 and len(lane_pred["pred_lanes"]) > 0:
            line_0 = rmse(gt_lane[0], pred_lane[0])
        if len(gt_lane) > 1 and len(lane_pred["pred_lanes"]) > 1:
            line_1 = rmse(gt_lane[1], pred_lane[1])
        return line_0, line_1

    def annotation_parser(self, annotations):
        lanes = [[],[]]
        angles = [[],[]]
        category = [-1,-1]
        colorOut = [-1,-1]
        for anno in annotations:
            label = self.category[anno['label_id']]
            if label == "condition":
                skyline = (anno["points"][1] + anno["points"][3]) / 2
                continue
            if int(anno['attributes']['ID']) not in [0, 1]:
                continue

            color = anno['attributes']['color']
            points = np.array(anno['points'], np.float32).reshape(-1, 2).tolist()
            org_lanes = sorted(points, key=lambda x: -x[-1])
            xp = []
            yp = []
            beginIdx = 0
            endIdx = 0
            org_lane = []
            for i in range(len(org_lanes) - 1, -1, -1):
                org_lane.append(org_lanes[i])

            for i in range(len(org_lane)):
                xp.append(int(org_lane[i][0]))
                yp.append(int(org_lane[i][1] - self.crop_a))

            if yp[0] <= self.h_samples[0] and yp[-1] <= self.h_samples[0]:
                continue

            if (len(xp) <= 1):  # 真值一个点：不参与比对
                continue

            angle = get_angle(np.array(xp), np.array(yp))

            for i in range(len(self.h_samples)):
                if self.h_samples[i] >= yp[0]:
                    beginIdx = i
                    break

            for i in range(len(self.h_samples)):
                if self.h_samples[i] >= yp[-1]:
                    endIdx = i
                    break

            if endIdx == 0:
                endIdx = len(self.h_samples)

            yvals = self.h_samples[beginIdx:endIdx]
            yp = np.array(yp)
            xp = np.array(xp)
            _, unique_indices = np.unique(yp, return_index=True)
            # 步骤 2: 提取唯一值（保留第一个出现的值）
            yp = yp[unique_indices]
            xp = xp[unique_indices]
            # 步骤 3: 重新排序（确保单调递增）
            sort_indices = np.argsort(yp)
            yp = yp[sort_indices]
            xp = xp[sort_indices]
            if len(yp) < 2:
                continue
            fun = interpolate.interp1d(yp, xp, kind="slinear")
            xinter = fun(yvals)

            xinterp = []
            for i in range(len(xinter)):
                xinterp.append(int(xinter[i]))

            beginV = []
            endV = []
            if beginIdx > endIdx:
                endIdx = beginIdx
            for i in range(len(self.h_samples)):
                if i < beginIdx:
                    beginV.append(-2)
                if i >= endIdx:
                    endV.append(-2)
            lane = beginV
            lane.extend(xinterp)
            lane.extend(endV)

            lanes[int(anno["attributes"]["ID"])] = lane
            # lanes.append(lane)
            angles[int(anno["attributes"]["ID"])] = angle
            # angles.append(angle)
            c = self.CATE_LIST.index(label)
            category[int(anno["attributes"]["ID"])] = c
            # category.append(c)  # c0+1
            # 暂时将橘色车道线和黄色车道线都归为黄色车道线
            if (color == 'orange'):
                color = 'yellow'
            colorOut[int(anno["attributes"]["ID"])] = color
            # colorOut.append(self.COLOR_LIST.index(color))
        if len(lanes[0]) == 0 or len(lanes[1]) == 0:
            lanes = []
            category = []
            colorOut = []
            angles = []
        return {'gt_lanes': lanes, "gt_cls": category, "gt_colors":colorOut,'gt_angles': angles, "gt_skyline":skyline}

    def pred_parser(self, pred_path):
        label = open(pred_path, 'r')
        category = []
        color = []
        lanes = []
        angles = []
        is_virtual = []
        order = 0
        # Load the annotation information
        for line in label:
            if order >= self.test_pre_num:
                continue
            lane = []
            line = line.strip()
            line = line.split()
            track_id = int(line[0])  # track_id=0的是虚拟线

            for i in range(7, len(line)):
                lane.append(int(line[i]))
            ###########多检的车道线且只有一个点去掉############
            lane_cs = []
            for i in range(len(lane)):
                if lane[i] != -2:
                    lane_cs.append(lane[i])
            if len(lane_cs) == 1:
                continue
            #############################################
            category.append(int(line[1]))  # 类型

            if (int(line[2]) == 2 or int(line[2]) == 4):  # 黄色和橙色
                color.append(self.COLOR_LIST.index("yellow"))  # 颜色
            elif (int(line[2]) == 1):
                color.append(self.COLOR_LIST.index("white"))  # 颜色
            elif (int(line[4]) == 3):
                color.append(self.COLOR_LIST.index("blue"))  # 3蓝色
            else:
                color.append(self.COLOR_LIST.index("other"))  # 未知 和其他

            angle = get_angle(np.array(lane), np.array(self.ys))

            lanes.append(lane)
            angles.append(angle)
            if track_id == 0:
                is_virtual.append(1)
            else:
                is_virtual.append(0)
            order += 1
        
        if len(lanes) < 2:
            lanes = []
            category = []
            colorOut = []
            angles = []
            is_virtual = []

        return {'pred_lanes': lanes, 'pred_cls': category,
                                      'pred_colors': color,
                                      'pred_angles': angles,
                                      'is_virtual': is_virtual}

    def img_parser(self, pred_path):
        img = cv2.imread(os.path.join(self.origin_img_path, os.path.basename(pred_path).replace(".txt", ".jpg")))
        img_w = img.shape[1]
        img_h = img.shape[0]
        img1 = img[self.GLOBAL_CROP_A:img_h, :].copy()
        return img1

    def Kps_acc_segm(self, gt_lanes, pred_lanes, distance, angles, skylineY):  # ,tp,tp_class,fn,fn_class):
        temp_fn = 0
        tempDist = []
        for i in range(self.MAX_DIS):
            tempDist.append(0)
        ave_dist = line_accuracy(pred_lanes, gt_lanes, angles, angles, skylineY, self.GLOBAL_CROP_A, self.input_size_h, self.R)
        if ave_dist < self.MAX_DIS:
            for k in range(self.MAX_DIS):
                if ave_dist == k:
                    tempDist[k] += 1
        else:
            temp_fn = 1
        if temp_fn == 0:
            for k in range(self.MAX_DIS):
                distance[k] += tempDist[k]

        return distance

    def acc_segm(self, lane_annotation, lane_pred, img):
        tp_0, fp_0, fn_0 = 0, 0, 0
        tp_1, fp_1, fn_1 = 0, 0, 0

        tp_cate_0, tp_cate_1 = [0] * len(self.CATE_LIST), [0] * len(self.CATE_LIST)
        fn_cate_0, fn_cate_1 = [0] * len(self.CATE_LIST), [0] * len(self.CATE_LIST)
        fp_cate_0, fp_cate_1 = [0] * len(self.CATE_LIST), [0] * len(self.CATE_LIST)
        cate_list_0, cate_list_1 = [0] * len(self.CATE_LIST), [0] * len(self.CATE_LIST)
        tp_color_0, tp_color_1 = [0] * len(self.CATE_LIST), [0] * len(self.CATE_LIST)
        color_list_0, color_list_1 = [0] * len(self.CATE_LIST), [0] * len(self.CATE_LIST)


        distance = []
        for i in range(self.MAX_DIS):
            distance.append(0)
        bp_distance = []
        for i in range(6):
            bp_distance.append(0)

        gt_lanes = lane_annotation['gt_lanes']
        gt_cls0 = lane_annotation['gt_cls']
        gt_colors = lane_annotation['gt_colors']
        pred_lanes = lane_pred['pred_lanes']
        pred_cls0 = lane_pred['pred_cls']
        pred_colors = lane_pred['pred_colors']
        skylineY = lane_annotation['gt_skyline']
        is_virtual = lane_pred['is_virtual'] # '2022-09-23-02-39-05-SH.MP4_0'
        is_contain = np.any(is_virtual)
        if is_contain:
            self.contain_virtual += 1
        num_virtual = np.count_nonzero(is_virtual)
        self.total_virtuals += num_virtual
        self.total_gt_lanes += len(gt_lanes)
        self.total_imgs += 1

        angles = lane_annotation['gt_angles']
        angles_pred = lane_pred['pred_angles']

        tempMask = []
        for j in range(len(pred_lanes)):
            tempMask.append(0)
        gt_tempMask = []
        for i in range(len(gt_lanes)):
            gt_tempMask.append(0)

        # 漏检fn + 正检 tp -  first
        for i in range(len(gt_lanes)):
            if gt_tempMask[i] == 1:
                continue
            minDist = 100
            maxindex = -1

            for j in range(len(pred_lanes)):
                if tempMask[j] == 1:
                    continue
                line_acc = line_accuracy(pred_lanes[j], gt_lanes[i], angles[i], angles_pred[j], skylineY, self.GLOBAL_CROP_A, self.input_size_h, self.R)

                if gt_cls0[i] == self.CATE_LIST.index('Roadline') or gt_cls0[i] == self.CATE_LIST.index('Fence'):
                    if pred_cls0[j] == self.CATE_LIST.index('Roadline') or pred_cls0[j] == self.CATE_LIST.index('Fence'):
                        if line_acc < self.MAX_DIS:
                            line_acc = -1
                if (gt_cls0[i] == self.CATE_LIST.index('Roadline') and pred_cls0[j] == self.CATE_LIST.index('Roadline')) or (gt_cls0[i] == self.CATE_LIST.index('Fence') and pred_cls0[j] == self.CATE_LIST.index('Fence')):
                    if line_acc < self.MAX_DIS:
                        line_acc = -2
                if line_acc < minDist and line_acc < self.MAX_DIS:
                    minDist = line_acc
                    maxindex = j

            gt_minDist = minDist
            gt_maxindex = i
            if maxindex > -1:
                for k in range(len(gt_lanes)):
                    if gt_tempMask[k] == 1 or gt_cls0[k] == 0:
                        continue
                    line_acc = line_accuracy(pred_lanes[maxindex], gt_lanes[k], angles[k], angles_pred[maxindex], skylineY, self.GLOBAL_CROP_A, self.input_size_h, self.R)

                    if gt_cls0[k] == self.CATE_LIST.index('Roadline') or gt_cls0[k] == self.CATE_LIST.index('Fence'):
                        if pred_cls0[maxindex] == self.CATE_LIST.index('Roadline') or pred_cls0[maxindex] == self.CATE_LIST.index('Fence'):
                            if line_acc < self.MAX_DIS:
                                line_acc = -1
                    if (gt_cls0[i] == self.CATE_LIST.index('Roadline') and pred_cls0[j] == self.CATE_LIST.index('Roadline')) or (gt_cls0[i] == self.CATE_LIST.index('Fence') and pred_cls0[j] == self.CATE_LIST.index('Fence')):
                        if line_acc < self.MAX_DIS:
                            line_acc = -2
                    if line_acc < gt_minDist:
                        gt_minDist = line_acc
                        gt_maxindex = k

                if gt_maxindex == i:
                    distance = self.Kps_acc_segm(gt_lanes[gt_maxindex], pred_lanes[maxindex], distance, angles[gt_maxindex], skylineY)
                    if gt_maxindex == 0:
                        tp_0 += 1
                    elif gt_maxindex == 1:
                        tp_1 += 1
                    else:
                        raise

                    tempMask[maxindex] = 1
                    gt_tempMask[gt_maxindex] = 1

                    for l in range(len(self.CATE_LIST)):
                        if gt_cls0[gt_maxindex] == l:
                            if gt_maxindex == 0:
                                tp_cate_0[l] += 1
                            elif gt_maxindex == 1:
                                tp_cate_1[l] += 1
                            else:
                                raise
                            if pred_cls0[maxindex] == l:
                                if maxindex == 0:
                                    cate_list_0[l] += 1
                                elif maxindex == 1:
                                    cate_list_1[l] += 1
                                else:
                                    raise

                    for l in range(len(self.COLOR_LIST)):
                        if gt_colors[gt_maxindex] == l:  # color 类型0,1,2.
                            if gt_maxindex == 0:
                                tp_color_0[l] += 1
                            elif gt_maxindex == 1:
                                tp_color_1[l] += 1
                            else:
                                raise
                            if (gt_colors[gt_maxindex] == pred_colors[maxindex]
                                    or gt_cls0[gt_maxindex] == self.CATE_LIST.index('Fence')
                                    or gt_cls0[gt_maxindex] == self.CATE_LIST.index('Roadline')
                                    or pred_cls0[maxindex] == self.CATE_LIST.index('Fence')
                                    or pred_cls0[maxindex] == self.CATE_LIST.index('Roadline')):
                                if gt_maxindex == 0:
                                    color_list_0[l] += 1
                                elif gt_maxindex == 1:
                                    color_list_1[l] += 1
                                else:
                                    raise
                    # draw
                    myDraw_line(img, gt_lanes[gt_maxindex], self.ys, self.tp_color)
                    myDraw_circle(img, pred_lanes[maxindex], self.ys, self.tp_color, is_virtual[maxindex])

        # 漏检fn + 正检 tp  _second
        for i in range(len(gt_lanes)):
            if gt_tempMask[i] == 1 or gt_cls0[i] == 0:
                continue
            minDist = 100
            maxindex = -1
            for j in range(len(pred_lanes)):
                if tempMask[j] == 1:
                    continue
                line_acc = line_accuracy(pred_lanes[j], gt_lanes[i], angles[i], angles_pred[j], skylineY, self.GLOBAL_CROP_A, self.input_size_h, self.R)

                if gt_cls0[i] == self.CATE_LIST.index('Roadline') or gt_cls0[i] == self.CATE_LIST.index('Fence'):
                    if pred_cls0[j] == self.CATE_LIST.index('Roadline') or pred_cls0[j] == self.CATE_LIST.index('Fence'):
                        if line_acc < self.MAX_DIS:
                            line_acc = -1
                if (gt_cls0[i] == self.CATE_LIST.index('Roadline') and pred_cls0[j] == self.CATE_LIST.index('Roadline')) or (gt_cls0[i] == self.CATE_LIST.index('Fence') and pred_cls0[j] == self.CATE_LIST.index('Fence')):
                    if line_acc < self.MAX_DIS:
                        line_acc = -2
                if line_acc < minDist and line_acc < self.MAX_DIS:
                    minDist = line_acc
                    maxindex = j

            gt_minDist = minDist
            gt_maxindex = i
            if maxindex > -1:
                if gt_maxindex == 0:
                    tp_0 += 1
                elif gt_maxindex == 1:
                    tp_1 += 1
                else:
                    raise

                tempMask[maxindex] = 1
                gt_tempMask[gt_maxindex] = 1

                for l in range(len(self.CATE_LIST)):
                    if gt_cls0[gt_maxindex] == l:
                        if gt_maxindex == 0:
                            tp_cate_0[l] += 1
                        elif gt_maxindex == 1:
                            tp_cate_1[l] += 1
                        else:
                            raise
                        if pred_cls0[maxindex] == l:
                            if maxindex == 0:
                                cate_list_0[l] += 1
                            elif maxindex == 1:
                                cate_list_1[l] += 1
                            else:
                                raise

                for l in range(len(self.COLOR_LIST)):
                    if gt_colors[gt_maxindex] == l:
                        if gt_maxindex == 0:
                            tp_color_0[l] += 1
                        elif gt_maxindex == 1:
                            tp_color_1[l] += 1
                        else:
                            raise
                        if (gt_colors[gt_maxindex] == pred_colors[maxindex]
                                or gt_cls0[gt_maxindex] == self.CATE_LIST.index('Fence')
                                or gt_cls0[gt_maxindex] == self.CATE_LIST.index('Roadline')
                                or pred_cls0[maxindex] == self.CATE_LIST.index('Fence')
                                or pred_cls0[maxindex] == self.CATE_LIST.index('Roadline')
                        ):
                            if maxindex == 0:
                                color_list_0[l] += 1
                            elif maxindex == 1:
                                color_list_1[l] += 1
                            else:
                                raise
                myDraw_line(img, gt_lanes[gt_maxindex], self.ys, self.tp_color)
                myDraw_circle(img, pred_lanes[maxindex], self.ys, self.tp_color, is_virtual[maxindex])
            else:
                maxIndex = -1
                if gt_cls0[i] != self.CATE_LIST.index('Roadline') and gt_cls0[i] != self.CATE_LIST.index( 'Fence'):  # 不是路沿围栏，但周围有路沿围栏
                    for k in range(len(gt_lanes)):
                        if i == k:
                            continue
                        if gt_cls0[k] != self.CATE_LIST.index('Roadline') and gt_cls0[k] != self.CATE_LIST.index('Fence'):
                            continue
                        line_acc = line_accuracy(gt_lanes[i], gt_lanes[k], angles[k], angles[i], skylineY, self.GLOBAL_CROP_A, self.input_size_h, self.R)
                        if line_acc < self.MAX_DIS * 1.2:
                            maxIndex = k
                else:  # 是路沿围栏
                    for k in range(len(gt_lanes)):
                        if i == k:
                            continue
                        if gt_tempMask[k] == 0:  # 路沿or围栏 存在匹配成功的相邻线
                            if gt_cls0[k] == 0:  # 该目标是忽略目标，判断其是否有匹配目标
                                t_flag = 0
                                for t in range(len(pred_lanes)):
                                    t_lane_acc = line_accuracy(pred_lanes[t], gt_lanes[k], angles[k], angles_pred[t], skylineY, self.GLOBAL_CROP_A, self.input_size_h, self.R)
                                    if t_lane_acc < self.MAX_DIS * 1.2:
                                        t_flag = 1
                                if t_flag == 0:
                                    continue  # 忽略目标没有匹配成功、返回
                            else:
                                continue  # 不是忽略目标，直接返回
                        line_acc = line_accuracy(gt_lanes[i], gt_lanes[k], angles[k], angles[i], skylineY, self.GLOBAL_CROP_A, self.input_size_h, self.R)
                        if line_acc < self.MAX_DIS * 1.2:
                            maxIndex = k

                if maxIndex == -1:
                    if i == 0:
                        fn_0 +=1
                    elif i == 1:
                        fn_1 += 1
                    else:
                        raise

                    for l in range(len(self.CATE_LIST)):
                        if gt_cls0[i] == l:
                            if i == 0:
                                fn_cate_0[l] += 1  # fn of this cate
                            elif i == 1:
                                fn_cate_1[l] += 1
                            else:
                                raise

                    myDraw_line(img, gt_lanes[i], self.ys, self.fn_color)

                # else:
                #     myDraw_line(img, gt_lanes[i], self.ys, self.fn_color)


        # 多检fp
        for j in range(len(pred_lanes)):
            if tempMask[j] == 1:
                continue
            maxIndex = -1
            for k in range(len(gt_lanes)):
                if (pred_cls0[j] != self.CATE_LIST.index('Roadline')  # 预测的结果不是路沿
                        and pred_cls0[j] != self.CATE_LIST.index('Fence')  # 预测的结果不是护栏 # 若检测目标是路沿，无论k是否已配对，均可计算距离
                        and gt_cls0[k] != self.CATE_LIST.index('Roadline')  # gt不是路沿
                        and gt_cls0[k] != self.CATE_LIST.index('Fence')  # gt不是护栏
                        and gt_tempMask[k] == 1  # gt匹配过
                        and gt_cls0[k] != 0):  # gt的类别不是none
                    continue
                line_acc = line_accuracy(pred_lanes[j], gt_lanes[k], angles[k], angles_pred[j], skylineY, self.GLOBAL_CROP_A, self.input_size_h, self.R)
                if line_acc < self.MAX_DIS or (gt_cls0[k] == 0 and line_acc <self.MAX_DIS * 1.2):
                    maxIndex = k

            if maxIndex == -1:
                if j == 0:
                    fp_0 += 1
                elif j == 1:
                    fp_1 += 1
                else:
                    raise

                for l in range(len(self.CATE_LIST)):
                    if pred_cls0[j] == l:
                        if j == 0:
                            fp_cate_0[l] += 1
                        elif j == 1:
                            fp_cate_1[l] += 1
                        else:
                            raise
                myDraw_circle(img, pred_lanes[j], self.ys, self.fp_color, is_virtual[j])
        return img,tp_0, fp_0, fn_0, tp_1, fp_1, fn_1

    def show(self, lane_annotation, lane_pred, img, img_name):
        for gt_lane in lane_annotation["gt_lanes"]:
            myDraw_line(img, gt_lane, self.ys, self.fn_color)
        for pred_lane in lane_pred["pred_lanes"]:
            myDraw_circle(img, pred_lane, self.ys, [0, 255, 0], is_virtual=False)
        # cv2.imshow("d", img)
        # cv2.waitKey(0)
        cv2.imwrite("/home/macan/Lucky_macan/dataset/mx/AutoTest_20250213/show/" + img_name, img)

    def _eval(self, pred_results_path):
        results = sorted(glob(pred_results_path + "/*.txt"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for idx, result_path in enumerate(results):
            logger.info(f"正在处理：{result_path}")
            frame_id = int(os.path.basename(result_path).split("_")[-1].split(".")[0])
            video_name = os.path.basename(result_path).replace("_" + str(frame_id) + ".txt", "")

            lane_annotation = self.annotation_parser(self.annotation[video_name][frame_id]["annotations"])
            lane_pred = self.pred_parser(result_path)
            img = self.img_parser(result_path)
            img = self.acc_segm(lane_annotation, lane_pred, img)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            logger.info(f"0号车道信息：tp: {self.tp_0} fp: {self.fp_0} fn: {self.fn_0} tp_cate: {self.tp_cate_0} fn_cate: {self.fn_cate_0} fp_cate: {self.fp_cate_0}")
            logger.info(f"1号车道信息：tp: {self.tp_1} fp: {self.fp_1} fn: {self.fn_1} tp_cate: {self.tp_cate_1} fn_cate: {self.fn_cate_1} fp_cate: {self.fp_cate_1}")
            logger.info("=" * 100)

    def middle_point_eval(self, lane_annotation, lane_pred, img):
        if len(lane_annotation["gt_lanes"]) == 0 or len(lane_pred["pred_lanes"]) == 0:
            return img, "null"
        gt_middle_point = [(i + j) // 2 if i > 0 and j > 0 else -2 for i, j in zip(lane_annotation["gt_lanes"][0], lane_annotation["gt_lanes"][1])]
        pred_middle_point = [(i + j) // 2  if i > 0 and j > 0 else -2 for i, j in zip(lane_pred["pred_lanes"][0], lane_pred["pred_lanes"][1])]
        # myDraw_line(img, gt_middle_point, self.ys, [255, 255, 255])
        # myDraw_circle(img, pred_middle_point, self.ys, [0, 0, 0], is_virtual = False)
        gt_center = [gt_middle_point[self.center_position], int(self.ys[self.center_position])]
        pred_center = [pred_middle_point[self.center_position], int(self.ys[self.center_position])]
        img = draw_cross(img, gt_center, 10, color = [0,255,0], circle_size = 5)
        img = draw_cross(img, pred_center, 10, color=[0, 0, 255], circle_size = 5)
        l = rmse(gt_center, pred_center)
        img = draw_text(img, str(l), ((gt_center[0] + pred_center[0]) // 2, pred_center[1]))
        return img, l

    def last_middle_point(self, lane_pred, last_lane_pred, img):
        if last_lane_pred is None or len(last_lane_pred["pred_lanes"]) == 0 or len(lane_pred["pred_lanes"]) == 0:
            return img, "null"
        pred_middle_point = [(i + j) // 2 if i > 0 and j > 0 else -2 for i, j in zip(lane_pred["pred_lanes"][0], lane_pred["pred_lanes"][1])]
        last_pred_middle_point = [(i + j) // 2  if i > 0 and j > 0 else -2 for i, j in zip(last_lane_pred["pred_lanes"][0], last_lane_pred["pred_lanes"][1])]
        pred_center = [pred_middle_point[self.center_position], int(self.ys[self.center_position])]
        last_pred_center = [last_pred_middle_point[self.center_position], int(self.ys[self.center_position])]
        img = draw_cross(img, last_pred_center, 5, color=[0, 255, 255], circle_size=3)
        l = rmse(pred_center, last_pred_center)
        img = draw_text(img, str(l), ((last_pred_center[0] + pred_center[0]) // 2, pred_center[1] + 15), color = [0, 255, 255])
        return img, l

    def _get_lane_uv(self, p_world_start, p_world_end):
        p_uv_start = self.ipm_trans_.get_pixel_uv(p_world_start, is_distort=True)
        p_uv_end = self.ipm_trans_.get_pixel_uv(p_world_end, is_distort=True)
        return (p_uv_start, p_uv_end)        

    def _get_freeze_lanes(self, lane_width):
        p0_world_start = [-(lane_width / 2), 0, 6]
        p0_world_end = [-(lane_width / 2), 0, 18]

        p1_world_start = [(lane_width / 2), 0, 6]
        p1_world_end = [(lane_width / 2), 0, 18]

        return [self._get_lane_uv(p0_world_start, p0_world_end), 
                self._get_lane_uv(p1_world_start, p1_world_end)]

    def add_freeze_lanes(self, lane_width=3.5):
        lanes = []
        category = []
        color = []
        angles = []
        is_virtual = []
         
        freeze_lanes = self._get_freeze_lanes(lane_width)
        for lane_points in freeze_lanes:
            org_lanes = sorted(lane_points, key=lambda x: -x[-1])
            xp = []
            yp = []
            beginIdx = 0
            endIdx = 0
            org_lane = []
            for i in range(len(org_lanes) - 1, -1, -1):
                org_lane.append(org_lanes[i])

            for i in range(len(org_lane)):
                xp.append(int(org_lane[i][0]))
                yp.append(int(org_lane[i][1] - self.crop_a))

            if yp[0] <= self.h_samples[0] and yp[-1] <= self.h_samples[0]:
                continue

            if (len(xp) <= 1):  # 真值一个点：不参与比对
                continue

            angle = get_angle(np.array(xp), np.array(yp))

            for i in range(len(self.h_samples)):
                if self.h_samples[i] >= yp[0]:
                    beginIdx = i
                    break

            for i in range(len(self.h_samples)):
                if self.h_samples[i] >= yp[-1]:
                    endIdx = i
                    break

            if endIdx == 0:
                endIdx = len(self.h_samples)

            yvals = self.h_samples[beginIdx:endIdx]
            yp = np.array(yp)
            xp = np.array(xp)
            _, unique_indices = np.unique(yp, return_index=True)
            # 步骤 2: 提取唯一值（保留第一个出现的值）
            yp = yp[unique_indices]
            xp = xp[unique_indices]
            # 步骤 3: 重新排序（确保单调递增）
            sort_indices = np.argsort(yp)
            yp = yp[sort_indices]
            xp = xp[sort_indices]
            if len(yp) < 2:
                continue
            fun = interpolate.interp1d(yp, xp, kind="slinear")
            xinter = fun(yvals)

            xinterp = []
            for i in range(len(xinter)):
                xinterp.append(int(xinter[i]))

            beginV = []
            endV = []
            if beginIdx > endIdx:
                endIdx = beginIdx
            for i in range(len(self.h_samples)):
                if i < beginIdx:
                    beginV.append(-2)
                if i >= endIdx:
                    endV.append(-2)
            lane = beginV
            lane.extend(xinterp)
            lane.extend(endV)
            
            lanes.append(lane)
            category.append(self.CATE_LIST.index('SingleDotted'))
            color.append(self.COLOR_LIST.index('white'))
            is_virtual.append(1)
            angles.append(angle)

        return {'pred_lanes': lanes, 'pred_cls': category,
                                      'pred_colors': color,
                                      'pred_angles': angles,
                                      'is_virtual': is_virtual}

    def eval(self, pred_results_path):
        results = sorted(glob(pred_results_path + "/*.txt"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        last_pred = None
        img_names = []
        tps = []
        tns = []
        fps = []
        fns = []
        line_0s = []
        line_1s = []
        tp_0_s = []
        tp_1_s = []
        fp_0_s = []
        fp_1_s = []
        fn_0_s = []
        fn_1_s = []
        ls = []
        lls = []

        for idx, result_path in enumerate(results):
            img_name = os.path.basename(result_path).replace(".txt", ".jpg")
            frame_id = int(os.path.basename(result_path).split("_")[-1].split(".")[0])
            video_name = os.path.basename(result_path).replace("_" + str(frame_id) + ".txt", "")
            lane_annotation = self.annotation_parser(self.annotation[video_name][frame_id]["annotations"])
            lane_pred = self.prediction[video_name][frame_id]
            img = self.img_parser(result_path)
            tp, fp, fn, tn = self._whole_lane_eval(lane_annotation, lane_pred)
            lane_pred = self.prediction_add_motan[video_name][frame_id]

            img, tp_0, fp_0, fn_0, tp_1, fp_1, fn_1 = self.acc_segm(lane_annotation, lane_pred, img)
            line_0, line_1 = self.lane_line_distance_eval(lane_annotation, lane_pred)
            img, l = self.middle_point_eval(lane_annotation, lane_pred, img)
            img, ll = self.last_middle_point(lane_pred, last_pred, img)
            img_names.append(img_name)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            tns.append(tn)
            line_0s.append(line_0)
            line_1s.append(line_1)
            tp_0_s.append(tp_0)
            tp_1_s.append(tp_1)
            fp_0_s.append(fp_0)
            fp_1_s.append(fp_1)
            fn_0_s.append(fn_0)
            fn_1_s.append(fn_1)
            ls.append(l)
            lls.append(ll)
            # logger.info(f"{os.path.basename(result_path)} | 0号车道线tp: {tp_0} 0号车道线fp: {fp_0} 0号车道线fn: {fn_0} "
            #             f"0号车道线贴合度: {line_0} 1号车道线tp: {tp_1} 1号车道线fp: {fp_1} 1号车道线fn: {fn_1} 1号车道线贴合度: {line_1} "
            #             f"当前帧中心点误差: {l} 与上一帧中心点误差: {ll} 车道tp: {tp} 车道fp: {fp} 车道fn: {fn} 车道tn: {tn}")
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

            save_path = os.path.join(self.img_save_path, video_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite( save_path + "/" + img_name, img)
            last_pred = lane_pred

        data = {
            '图片名称': img_names,
            "车道TP": tps,
            "车道FP": fps,
            "车道FN": fns,
            "车道TN": tns,
            "0号车道线贴合度": line_0s,
            "1号车道线贴合度": line_1s,
            "标签与当前帧结果抖动": ls,
            "当前帧与上一帧抖动": lls,
            '0号车道线TP': tp_0_s,
            '1号车道线TP': tp_1_s,
            '0号车道线FP': fp_0_s,
            '1号车道线FP': fp_1_s,
            '0号车道线FN': fn_0_s,
            '1号车道线FN': fn_1_s
        }
        df = pd.DataFrame(data)

        # 保存为 CSV 文件        
        csv_file = os.path.join(args.debug_path, f'{os.path.basename(pred_results_path)}.csv')
        df.to_csv(csv_file, index=False)

        logger.info(f'保存统计文件：{csv_file}')

    def process(self, video_name):
        logger.info(f"Processing video {video_name}")
        pred_results_path = os.path.join(self.pred_path, video_name)
        self.eval(pred_results_path)

def parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pred_path', type=str, help='预测结果路径', default="/home/lc/work/sdk/dynamic_test/AutoTest_20250213_origin")
    # parser.add_argument('--pred_path', type=str, help='预测结果路径', default="/home/lc/work/sdk/dynamic_test/AutoTest_20250213_6.1.15_SOP2/AutoTest_20250213_6.1.15_processed")
    parser.add_argument('--pred_path', type=str, help='预测结果路径', default="/home/lc/work/sdk/dynamic_test/v7.1.1_processed")
    parser.add_argument('--annotation_path', type=str, default='/home/lc/work/sdk/dynamic_test/version3/annotations/default.json', help='真值标注路径')
    parser.add_argument('--image_path', type=str, default='/home/lc/work/sdk/dynamic_test/together', help='输入图片所在目录')
    # parser.add_argument('--debug_path', type=str, default='./debug_20250707', help='可视化图片保存路径')
    # parser.add_argument('--debug_path', type=str, default='./debug_sop2', help='可视化图片保存路径')
    # parser.add_argument('--debug_path', type=str, default='./debug_sop3', help='可视化图片保存路径')
    parser.add_argument('--debug_path', type=str, default='./debug_v7.1.1', help='可视化图片保存路径')
    args = parser.parse_args()
    logger.info(args)
    return args

if __name__ == '__main__':
    args = parser()
    m = Metric(image_path=args.image_path,
               annotation_path=args.annotation_path,
               pred_path=args.pred_path,
               img_save_path=args.debug_path)
    # 11-04-47-2024-11-14-Nor_000120_000134
    # 12-52-47-2024-11-14-Nor_000230_000243
    # 14-19-32-2024-11-26-Nor_000000_000123
    m.process(video_name="11-04-47-2024-11-14-Nor_000120_000134")
    m.process(video_name="12-52-47-2024-11-14-Nor_000230_000243")
    m.process(video_name="14-19-32-2024-11-26-Nor_000000_000123")
    