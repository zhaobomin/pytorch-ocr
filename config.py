#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# ============CRNN OCR模型配置===========#
ocrType = 'chinese'
ocrCfgPath = 'weights/ocr/{}/ocr.cfg'.format(ocrType)
#ocrPath = 'weights/ocr/{}/ocr.weights'.format(ocrType)
ocrPath = 'weights/ocr/{}/ocr_best.pth'.format(ocrType)
charactersPath = 'weights/ocr/chinese/ocr.json'


def get_characters():
    with open(charactersPath, encoding='utf-8') as f:
        characters = json.loads(f.read())
        characters = ' '+characters+'-｜'

        return characters, len(characters)


# ============CTPN TEXT 模型配置===========#
textCfgPath = 'weights/text/text.cfg'
textPath = 'weights/text/text.pth'

TEXT_LINE_SCORE = 0.7  # text line prob
scale = 600
maxScale = 900

# ============CTPN TEXT训练参数===========#
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

OHEM = True
RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300
