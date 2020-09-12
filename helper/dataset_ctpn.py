# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from helper.image import cal_rpn
from PIL import Image
from helper.image import resize_img
from torch.autograd import Variable


'''
从xml文件中读取图像中的真值框
'''


def readxml(path):
    gtboxes = []
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    xmin = int(round(float(attr.find('xmin').text)))
                    ymin = int(round(float(attr.find('ymin').text)))
                    xmax = int(round(float(attr.find('xmax').text)))
                    ymax = int(round(float(attr.find('ymax').text)))
                    gtboxes.append((xmin, ymin, xmax, ymax))

    return np.array(gtboxes)


'''
读取VOC格式数据，返回用于训练的图像、anchor目标框、标签
'''


class VOCDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)

    def generate_gtboxes(self, xml_path, rescale_fac=1.0):
        base_gtboxes = readxml(xml_path)
        gtboxes = []
        for base_gtbox in base_gtboxes:
            xmin, ymin, xmax, ymax = base_gtbox
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                next = 16*i-0.5
                gtboxes.append((prev, ymin, next, ymax))
                prev = next
            gtboxes.append((prev, ymin, xmax, ymax))
        return np.array(gtboxes)

    def __getitem__(self, idx, scale=600, maxScale=900):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        img = Image.open(img_path)

        img, rate = resize_img(img, scale, maxScale=maxScale)
        rescale_fac = 1/rate

        h, w = img.shape[:2]
        img = cv2.dnn.blobFromImage(
            img, scalefactor=1.0, size=(w, h), swapRB=False, crop=False)
        img = img.reshape(3, h, w)

        # = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        img = torch.tensor(img)

        #img = Variable(img).type(Tensor)
        xml_path = os.path.join(self.labelsdir, img_name.split('.')[0]+'.xml')
        gtbox = self.generate_gtboxes(xml_path, rescale_fac)
        feature_size = (int(np.ceil(h/16)), int(np.ceil(w/16)))
        [cls, regr] = cal_rpn((h, w), feature_size, 16, gtbox)
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        cls = np.expand_dims(cls, axis=0)

        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return img, cls, regr


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
    # x = cx-w/2
    # y = cy-h/2
    sinA = (h*(x1-cx)-w*(y1 - cy))*1.0/(h*h+w*w)*2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def draw_frame(frame, objects):
    origin_im_size = frame.shape[:-1]

    for box in objects:
        # Validation bbox of detected object
        # if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
        #    continue

        x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
        degree, w, h, x_center, y_center = solve(box)
        xmin = int(min(box[0::2]))
        xmax = int(max(box[0::2]))
        ymin = int(min(box[1::2]))
        ymax = int(max(box[1::2]))
        color = (255, 0, 0)
        frame = np.array(frame)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    return frame


def rotate_cut_img(im, box, leftAdjust=0.0, rightAdjust=0.0):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    degree, w, h, x_center, y_center = solve(box)
    xmin_ = min(box[0::2])
    xmax_ = max(box[0::2])
    ymin_ = min(box[1::2])
    ymax_ = max(box[1::2])
    x_center = x_center-xmin_
    y_center = y_center-ymin_
    im = im.crop([xmin_, ymin_, xmax_, ymax_])
    degree_ = degree*180.0/np.pi
    xmin = max(1, x_center-w/2-leftAdjust*(w/2))
    ymin = y_center-h/2
    xmax = min(x_center+w/2+rightAdjust*(w/2), im.size[0]-1)
    ymax = y_center+h/2
    newW = xmax-xmin
    newH = ymax-ymin
    tmpImg = im.rotate(degree_, center=(x_center, y_center)
                       ).crop([xmin, ymin, xmax, ymax])
    box = {'cx': x_center+xmin_, 'cy': y_center +
           ymin_, 'w': newW, 'h': newH, 'degree': degree_}
    return tmpImg
