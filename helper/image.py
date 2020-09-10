#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image
@author: chineseocr
"""
import numpy as np
import cv2
import requests
import six
from PIL import Image
import traceback
import base64
import datetime as dt
from config import *


def get_now():
    """
    获取当前时间
    """
    try:
        now = dt.datetime.now()
        nowString = now.strftime('%Y-%m-%d %H:%M:%S')
    except:
        nowString = '00-00-00 00:00:00'
    return nowString


def read_url_img(url):
    """
    爬取网页图片
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Safari/537.36'}
    try:
        req = requests.get(url, headers=headers, timeout=5)  # 访问时间超过5s，则超时
        if req.status_code == 200:
            imgString = req.content
            buf = six.BytesIO()
            buf.write(imgString)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            return img
        else:
            return None
    except:
        # traceback.print_exc()
        return None


def base64_to_PIL(string):
    try:

        base64_data = base64.b64decode(string.split('base64,')[-1])
        buf = six.BytesIO()
        buf.write(base64_data)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img
    except:
        return None


def soft_max(x):
    """numpy softmax"""
    expz = np.exp(x)
    sumz = np.sum(expz, axis=1)
    return expz[:, 1]/sumz


def reshape_tensor(x):
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(x.size(0), x.size(1)*x.size(2)*10, 2)
    return x


def reshape(x):
    b = x.shape
    x = x.transpose(0, 2, 3, 1)
    b = x.shape
    x = np.reshape(x, [b[0], b[1]*b[2]*10, 2])
    return x


def resize_img(image, scale, maxScale=None):
    """
    image :BGR array 
    """
    image = np.copy(image)
    vggMeans = [122.7717, 102.9801, 115.9465]
    imageList = cv2.split(image.astype(np.float32))
    imageList[0] = imageList[0]-vggMeans[0]
    imageList[1] = imageList[1]-vggMeans[1]
    imageList[2] = imageList[2]-vggMeans[2]
    image = cv2.merge(imageList)
    h, w = image.shape[:2]
    rate = scale/min(h, w)
    if maxScale is not None:
        if rate*max(h, w) > maxScale:
            rate = maxScale/max(h, w)

    image = cv2.resize(image, None, None, fx=rate, fy=rate,
                       interpolation=cv2.INTER_LINEAR)
    #image = cv2.resize(image, (608, 288), interpolation=cv2.INTER_LINEAR)
    return image, rate


'''
anchor生成
遇到的问题：首先，base_anchor 为初始位置点生成的anchor，按步长在feature map 的各个点生成anchor之后，anchors的 shape 为[10, h*w, 4]。
这里，我一开始是直接将anchors reshape 成 [10*h*w, 4]，这在训练时不收敛。
原因浅析：按我代码的实现方式，直接[10, h*w, 4] -> [10*h*w, 4]，anchor 的排列顺序将按照不同的anchor形状（共10种）进行排列，而不是根据feature map 的点按序排列，
而按 ctpn 的实现方式，小的anchor需要连成大的文本框才是最终的结果，不按点的顺序生成anchor可能给训练带来较大的干扰。
解决方案：将 anchor 根据feature_map 的各个点，按序生成10个anchor重新排列，也即：[10, h*w, 4] -> [h*w, 10, 4] -> [10*h*w, 4]，问题解决。
'''


def gen_anchor(featuresize, scale,
               heights=[11, 16, 23, 33, 48, 68, 97, 139, 198, 283],
               widths=[16, 16, 16, 16, 16, 16, 16, 16, 16, 16]):
    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel()), axis=1)

    # base center(x,,y) -> (x1, y1, x2, y2)
    base_anchor = np.array([0, 0, 15, 15])
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    anchor = list()
    for i in range(base_anchor.shape[0]):
        anchor_x1 = shift[:, 0] + base_anchor[i][0]
        anchor_y1 = shift[:, 1] + base_anchor[i][1]
        anchor_x2 = shift[:, 2] + base_anchor[i][2]
        anchor_y2 = shift[:, 3] + base_anchor[i][3]
        anchor.append(np.dstack((anchor_x1, anchor_y1, anchor_x2, anchor_y2)))

    return np.squeeze(np.array(anchor)).transpose((1, 0, 2)).reshape((-1, 4))


'''
anchor 与 bbox的 iou计算
iou = inter_area/(bb_area + anchor_area - inter_area)
'''


def compute_iou(anchors, bbox):
    ious = np.zeros((len(anchors), len(bbox)), dtype=np.float32)
    anchor_area = (anchors[:, 2] - anchors[:, 0]) * \
        (anchors[:, 3] - anchors[:, 1])
    for num, _bbox in enumerate(bbox):
        bb = np.tile(_bbox, (len(anchors), 1))
        bb_area = (bb[:, 2] - bb[:, 0])*(bb[:, 3] - bb[:, 1])
        inter_h = np.maximum(np.minimum(
            bb[:, 3], anchors[:, 3]) - np.maximum(bb[:, 1], anchors[:, 1]), 0)
        inter_w = np.maximum(np.minimum(
            bb[:, 2], anchors[:, 2]) - np.maximum(bb[:, 0], anchors[:, 0]), 0)
        inter_area = inter_h*inter_w
        ious[:, num] = inter_area/(bb_area + anchor_area - inter_area)

    return ious


'''
计算 anchor与 gtboxes在垂直方向的差异参数 regression_factor(Vc, Vh)
1、(x1, y1, x2, y2) -> (ctr_x, ctr_y, w, h)
2、 Vc = (gt_y - anchor_y) / anchor_h
    Vh = np.log(gt_h / anchor_h)
'''


def bbox_transfrom(anchors, gtboxes):
    gt_y = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    gt_h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0

    anchor_y = (anchors[:, 1] + anchors[:, 3]) * 0.5
    anchor_h = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (gt_y - anchor_y) / anchor_h
    Vh = np.log(gt_h / anchor_h)

    return np.vstack((Vc, Vh)).transpose()


'''
已知 anchor和差异参数 regression_factor(Vc, Vh),计算目标框 bbox
'''


def transform_bbox(anchor, regression_factor):
    anchor_y = (anchor[:, 1] + anchor[:, 3]) * 0.5
    anchor_x = (anchor[:, 0] + anchor[:, 2]) * 0.5
    anchor_h = anchor[:, 3] - anchor[:, 1] + 1

    Vc = regression_factor[0, :, 0]
    Vh = regression_factor[0, :, 1]

    bbox_y = Vc * anchor_h + anchor_y
    bbox_h = np.exp(Vh) * anchor_h

    x1 = anchor_x - 16 * 0.5
    y1 = bbox_y - bbox_h * 0.5
    x2 = anchor_x + 16 * 0.5
    y2 = bbox_y + bbox_h * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox


'''
bbox 边界裁剪
    x1 >= 0
    y1 >= 0
    x2 < im_shape[1]
    y2 < im_shape[0]
'''


def clip_bbox(bbox, im_shape):
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


'''
bbox尺寸过滤，舍弃小于设定最小尺寸的bbox
'''


def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep


'''
RPN module
1、生成anchor
2、计算anchor 和真值框 gtboxes的 iou
3、根据 iou，给每个anchor分配标签，0为负样本，1为正样本，-1为舍弃项
    (1) 对每个真值框 bbox，找出与其 iou最大的 anchor，设为正样本
    (2) 对每个anchor，记录其与每个bbox求取的 iou中最大的值 max_overlap
    (3) 对max_overlap 大于设定阈值的anchor，将其设为正样本，小于设定阈值，则设定为负样本
4、过滤超出边界的anchor框，将其标签设定为 -1
5、选取不超过设定数量的正负样本
6、求取anchor 取得max_overlap 时的gtbbox之间的真值差异量(Vc, Vh)
'''


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    base_anchor = gen_anchor(featuresize, scale)
    overlaps = compute_iou(base_anchor, gtboxes)

    gt_argmax_overlaps = overlaps.argmax(axis=0)
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(
        overlaps.shape[0]), anchor_argmax_overlaps]

    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)
    labels[gt_argmax_overlaps] = 1
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0

    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgsize[1]) |
        (base_anchor[:, 3] >= imgsize[0])
    )[0]
    labels[outside_anchor] = -1

    fg_index = np.where(labels == 1)[0]
    if (len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(
            fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1
    if not OHEM:
        bg_index = np.where(labels == 0)[0]
        num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
        if (len(bg_index) > num_bg):
            labels[np.random.choice(bg_index, len(
                bg_index) - num_bg, replace=False)] = -1

    bbox_targets = bbox_transfrom(
        base_anchor, gtboxes[anchor_argmax_overlaps, :])

    return [labels, bbox_targets]


def get_origin_box(size, boxes, scale=16):
    w, h = size
    gridbox = gen_anchor((int(np.ceil(h/scale)), int(np.ceil(w/scale))), scale)
    #print('gridbox', gridbox.shape)

    gridcy = (gridbox[:, 1]+gridbox[:, 3])/2.0
    gridh = (gridbox[:, 3]-gridbox[:, 1]+1)

    cy = boxes[:, 0]*gridh+gridcy
    ch = np.exp(boxes[:, 1])*gridh

    ymin = cy-ch/2
    ymax = cy+ch/2
    gridbox[:, 1] = ymin
    gridbox[:, 3] = ymax
    #print('gridbox:', gridbox.shape)
    return gridbox


def nms(boxes, scores, score_threshold=0.5, nms_threshold=0.3):
    def box_to_center(box):
        xmin, ymin, xmax, ymax = box
        w = xmax-xmin
        h = ymax-ymin
        return [round(xmin, 4), round(ymin, 4), round(w, 4), round(h, 4)]

    newBoxes = [box_to_center(box) for box in boxes]
    newscores = [round(float(x), 6) for x in scores]
    index = cv2.dnn.NMSBoxes(
        newBoxes, newscores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    if len(index) > 0:
        index = index.reshape((-1,))
        return boxes[index], scores[index]
    else:
        return [], []


def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)
    """

    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1+x3+x2+x4)/4.0
    cy = (y1+y3+y4+y2)/4.0
    w = (np.sqrt((x2-x1)**2+(y2-y1)**2)+np.sqrt((x3-x4)**2+(y3-y4)**2))/2
    h = (np.sqrt((x2-x3)**2+(y2-y3)**2)+np.sqrt((x1-x4)**2+(y1-y4)**2))/2
    sinA = (h*(x1-cx)-w*(y1 - cy))*1.0/(h*h+w*w)*2
    if abs(sinA) > 1:
        angle = None
    else:
        angle = np.arcsin(sinA)

    return angle, w, h, cx, cy


def rotate_nms(boxes, scores, score_threshold=0.5, nms_threshold=0.3):
    """
    boxes.append((center, (w,h), angle * 180.0 / math.pi))

    """
    def rotate_box(box):
        angle, w, h, cx, cy = solve(box)
        angle = round(angle, 4)
        w = round(w, 4)
        h = round(h, 4)
        cx = round(cx, 4)
        cy = round(cy, 4)
        return ((cx, cy), (w, h), angle)

    if len(boxes) > 0:
        newboxes = [rotate_box(box) for box in boxes]
        newscores = [round(float(x), 6) for x in scores]
        index = cv2.dnn.NMSBoxesRotated(
            newboxes, newscores, score_threshold=score_threshold, nms_threshold=nms_threshold)

        if len(index) > 0:
            index = index.reshape((-1,))
            return boxes[index], scores[index]
        else:
            return [], []
    else:
        return [], []


def get_boxes(bboxes):
    """
        boxes: bounding boxes
    """
    text_recs = np.zeros((len(bboxes), 8), np.int)
    index = 0
    for box in bboxes:

        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2

        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX*disX + disY*disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1*disX / width)
        y = np.fabs(fTmp1*disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y

        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
        index = index + 1

    boxes = []
    for box in text_recs:
        x1, y1 = (box[0], box[1])
        x2, y2 = (box[2], box[3])
        x3, y3 = (box[6], box[7])
        x4, y4 = (box[4], box[5])
        boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
    boxes = np.array(boxes)
    return boxes
