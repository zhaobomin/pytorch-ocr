
import config
from model_crnn import crnn_model
from model_ctpn import text_model
import sys
import cv2
import time
from PIL import Image

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    net_text = text_model.ctpn_net(
        'weights/text/text.pth', 'weights/text/text.cfg')
    net_ocr = crnn_model.crnn_net(
        config.ocrPath, config.ocrCfgPath, config.charactersPath)

    boxes, scores = net_text.predict(img)

    # print(boxes.shape)

    im = Image.fromarray(img)
    result = []
    for i, box in enumerate(boxes):
        if scores[i] > config.TEXT_LINE_SCORE:
            tmpImg = net_text.rotate_cut_img(
                im, box, leftAdjust=0.01, rightAdjust=0.01)
            text = net_ocr.predict(tmpImg)
            if text['text'] != '':
                text['box'] = [int(x) for x in box]
                text['textprob'] = round(float(scores[i]), 2)
                print(text['text'], text['box'], text['textprob'])
                result.append(text)

    img = net_text.draw_boxes(img, boxes)
    box_img_path = "test/pred_box.jpg"
    cv2.imwrite(box_img_path, img)
    print('save text box detect result:', box_img_path)
    result = sorted(result, key=lambda x: sum(x['box'][1::2]))
    # print(result)
