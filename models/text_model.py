#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from helper.dataset_ctpn import draw_frame, rotate_cut_img
import cv2
import numpy as np
import config
from helper.image import resize_img, get_origin_box, reshape_tensor
from helper.detectors import TextDetector
from models import darknet_model
from PIL import Image
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class TEXT_NET:
    def __init__(self, weights=None, cfg=None):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        if cfg == None:
            cfg = config.textCfgPath
        self.net = darknet_model.Darknet(cfg).to(self.device)

        if weights == None:
            weights = config.textPath

        if weights.endswith('.weights'):
            self.net.load_darknet_weights(weights)
        else:
            self.net.load_state_dict(torch.load(
                weights, map_location=self.device))
        print('loading weights:%s Done' % weights)
        self.net.eval()

    def _detect_box(self, image, scale=600, maxScale=900):

        image, rate = resize_img(image, scale, maxScale=maxScale)

        h, w = image.shape[:2]
        image = cv2.dnn.blobFromImage(
            image, scalefactor=1.0, size=(w, h), swapRB=False, crop=False)

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        image = torch.Tensor(image)
        image = Variable(image).type(Tensor)

        res = self.net(image)

        out = res.detach().cpu().numpy()

        clsOut = reshape_tensor(res[:, :20, ...])
        boxOut = reshape_tensor(res[:, 20:, ...])

        boxOut = boxOut.detach().cpu().numpy()
        boxes = get_origin_box((w, h), boxOut[0])

        clsOut_softmax = F.softmax(clsOut[0], dim=-1)[:, 1]
        scores = clsOut_softmax.detach().cpu().numpy()

        # boxes越界的修正
        boxes[:, 0:4][boxes[:, 0:4] < 0] = 0
        boxes[:, 0][boxes[:, 0] >= w] = w-1
        boxes[:, 1][boxes[:, 1] >= h] = h-1
        boxes[:, 2][boxes[:, 2] >= w] = w-1
        boxes[:, 3][boxes[:, 3] >= h] = h-1

        return scores, boxes, rate, w, h

    def predict(self, image, scale=600,
                maxScale=900,
                MAX_HORIZONTAL_GAP=30,
                MIN_V_OVERLAPS=0.6,
                MIN_SIZE_SIM=0.6,
                TEXT_PROPOSALS_MIN_SCORE=0.7,
                TEXT_PROPOSALS_NMS_THRESH=0.3,
                TEXT_LINE_NMS_THRESH=0.9,
                TEXT_LINE_SCORE=0.9
                ):
        MAX_HORIZONTAL_GAP = max(16, MAX_HORIZONTAL_GAP)
        detectors = TextDetector(
            MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)
        scores, boxes, rate, w, h = self._detect_box(image, scale, maxScale)
        size = (h, w)
        text_lines, scores = detectors.detect(boxes, scores, size,
                                              TEXT_PROPOSALS_MIN_SCORE, TEXT_PROPOSALS_NMS_THRESH, TEXT_LINE_NMS_THRESH, TEXT_LINE_SCORE)
        if len(text_lines) > 0:
            text_lines = text_lines/rate
        return text_lines, scores

    def draw_boxes(self, img, boxes):
        img = draw_frame(img, boxes)
        return img

    def rotate_cut_img(self, im, box, leftAdjust=0.0, rightAdjust=0.0):
        return rotate_cut_img(im, box, leftAdjust, rightAdjust)


'''
回归损失: smooth L1 Loss
只针对正样本求取回归损失
L = 0.5*x**2  |x|<1
L = |x| - 0.5
sigma: 平滑系数
1、从预测框p和真值框g中筛选出正样本
2、|x| = |g - p|
3、求取loss，这里设置了一个平滑系数 1/sigma
  (1) |x|>1/sigma: loss = |x| - 0.5/sigma
  (2) |x|<1/sigma: loss = 0.5*sigma*|x|**2
'''


class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device
        self.L_regr = nn.SmoothL1Loss(size_average=True)

    # input = [N, anchor_nums, 2],target = [N, anchor_nums, 3]
    def forward(self, input, target):
        try:
            # gt的cls label
            cls = target[0, :, 0]
            # gt的boxes
            regression = target[0, :, 1:3]

            # 保留正样本的下标
            regr_keep = (cls == 1).nonzero()[:, 0]
            # 修改为 cls != -1：
            #regr_keep = (cls != -1).nonzero()[:, 0]

            # 正样本的box
            regr_true = regression[regr_keep]

            # 正样本位置，预测的box
            regr_pred = input[0][regr_keep]

            ####
            #loss = self.L_regr(regr_true, regr_pred)
            # return loss
            ####

            diff = torch.abs(regr_true - regr_pred)

            less_one = (diff < 1.0/self.sigma).float()
            loss_1 = less_one * 0.5 * (diff ** 2) * \
                self.sigma                               # diff < 1/sigma
            loss_2 = torch.abs(1 - less_one) * \
                (diff - 0.5 / self.sigma)                # diff >= 1/sigma

            loss = loss_1 + loss_2

            # diff_vh 相加
            loss = torch.sum(loss, 1)

            # diff_vc 求平均
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)

        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            loss = torch.tensor(0.0)

        return loss.to(self.device)


'''
分类损失: softmax loss
1、OHEM模式
  (1) 筛选出正样本，求取softmaxloss
  (2) 求取负样本数量N_neg, 指定样本数量N, 求取负样本的topK loss, 其中K = min(N_neg, N - len(pos_num))
  (3) loss = loss1 + loss2
2、求取NLLLoss，截断在(0, 10)区间
'''


class RPN_CLS_Loss(nn.Module):
    def __init__(self, device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device
        self.L_cls = nn.CrossEntropyLoss(reduction='none')

    # input = [N, anchor_nums, 2], target = [N, 1, anchor_nums]

    def forward(self, input, target):
        if config.OHEM:
            # gt的cls的scores
            cls_gt = target[0][0]

            num_pos = 0
            loss_pos_sum = 0

            # 正样本的loss
            if len((cls_gt == 1).nonzero()) != 0:
                cls_pos = (cls_gt == 1).nonzero()[:, 0]
                # gt的正样本score
                gt_pos = cls_gt[cls_pos].long()
                # gt的正样本的预测的score
                cls_pred_pos = input[0][cls_pos]

                loss_pos = self.L_cls(
                    cls_pred_pos.view(-1, 2), gt_pos.view(-1))
                loss_pos_sum = loss_pos.sum()
                num_pos = len(loss_pos)

            # 负样本 loss
            cls_neg = (cls_gt == 0).nonzero()[:, 0]
            gt_neg = cls_gt[cls_neg].long()
            cls_pred_neg = input[0][cls_neg]
            # 负样本loss
            loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1))
            # 取TopK的负样本loss
            loss_neg_topK, _ = torch.topk(loss_neg, min(
                len(loss_neg), config.RPN_TOTAL_NUM - num_pos))

            num_neg_pos = len(loss_neg_topK)

            loss_cls = loss_pos_sum/num_pos + loss_neg_topK.mean()

            #loss_cls = loss_cls / config.RPN_TOTAL_NUM
            #loss_cls = loss_cls / total_num

            return loss_cls.to(self.device)
        else:
            # gt的cls的scores
            y_true = target[0][0]
            # 去掉gt负样本后的位置
            cls_keep = (y_true != -1).nonzero()[:, 0]

            # 去掉gt负样本后的位置后的scores
            cls_true = y_true[cls_keep].long()

            # 去掉gt负样本后的位置后的预测的scores
            cls_pred = input[0][cls_keep]

            # loss = F.cross_entropy(cls_pred, cls_true)
            # original is sparse_softmax_cross_entropy_with_logits
            loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)
            loss = torch.clamp(torch.mean(loss), 0,
                               10) if loss.numel() > 0 else torch.tensor(0.0)

            return loss.to(self.device)
