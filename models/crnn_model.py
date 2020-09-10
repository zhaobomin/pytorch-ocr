#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models import darknet_model
import cv2
from torch.autograd import Variable
import os
import time
import json
import numpy as np
from PIL import Image
import torch
import config
import sys


class CRNN_NET:
    def __init__(self, weights=None, cfg=None):

        self.charactersPred, _ = config.get_characters()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if cfg == None:
            cfg = config.ocrCfgPath
        self.net = darknet_model.Darknet(cfg).to(self.device)

        if weights == None:
            weights = config.ocrPath
        if weights.endswith('.weights'):
            self.net.load_darknet_weights(weights)
        else:
            self.net.load_state_dict(torch.load(weights))
        print('loading weights:%s Done' % weights)
        self.net.eval()

    def _decode(self, pred):
        t = pred.argmax(axis=1)
        prob = [pred[ind, pb] for ind, pb in enumerate(t)]

        length = len(t)
        charList = []
        probList = []
        n = len(self.charactersPred)
        for i in range(length):
            if t[i] not in [n-1, n-1] and (not (i > 0 and t[i - 1] == t[i])):
                charList.append(self.charactersPred[t[i]])
                probList.append(prob[i])
        '''
        res = {'text': ''.join(charList),
               "prob": round(float(min(probList)), 2) if len(probList) > 0 else 0,
               "chars": [{'char': char, 'prob': round(float(p), 2)}for char, p in zip(charList, probList)]}
        '''
        res = {'text': ''.join(charList),
               "prob": float(min(probList)) if len(probList) > 0 else 0}
        return res

    def predict(self, image):
        scale = image.size[1]*1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        image = image.resize((w, 32), Image.BILINEAR)
        image = (np.array(image.convert('L'))/255.0-0.5)/0.5

        h, w = image.shape
        if w < 8:
            return {'chars': [], 'text': '', 'prob': 0}
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        image = Variable(torch.Tensor(image).view(
            1, 1, image.shape[0], image.shape[1]).type(Tensor))
        with torch.no_grad():
            y_pred = self.net(image)

        out = y_pred[0][:, 0, :].detach().numpy()
        out = out.transpose((1, 0))
        out = darknet_model.softmax(out)

        return self._decode(out)
