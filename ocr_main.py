import config
from models import crnn_model, text_model
import sys
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image


class OCR:
    def __init__(self):
        self.net_text = text_model.TEXT_NET()
        self.net_ocr = crnn_model.CRNN_NET()

    def predict_text(self, img, output_file=''):
        #img = Image.fromarray(img)
        boxes, scores = self.net_text.predict(img)
        if output_file != '':
            img_box = self.net_text.draw_boxes(img, boxes)
            cv2.imwrite(output_file, img_box)
        return boxes, scores

    def predict_ocr(self, img):
        img = Image.fromarray(img)
        res = self.net_ocr.predict(img)  # img = Image.open('test/dd.jpg')
        return res

    def predict(self, img, output_file=''):
        boxes, scores = self.predict_text(img, output_file)
        im = Image.fromarray(img)
        result = []
        for i, box in enumerate(boxes):
            if scores[i] > config.TEXT_LINE_SCORE:
                tmpImg = self.net_text.rotate_cut_img(
                    im, box, leftAdjust=0.01, rightAdjust=0.01)
                text = self.net_ocr.predict(tmpImg)
                if text['text'] != '':
                    text['box'] = [int(x) for x in box]
                    text['textprob'] = round(float(scores[i]), 2)
                    #print(text['text'], text['box'], text['textprob'])
                    result.append(text)

        result = sorted(result, key=lambda x: sum(x['box'][1::2]))
        return result
