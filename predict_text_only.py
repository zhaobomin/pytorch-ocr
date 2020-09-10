import sys
import cv2
from models import text_model
if __name__ == '__main__':

    img = cv2.imread(sys.argv[1])
    net = text_model.TEXT_NET()

    boxes, scores = net.predict(img)
    img = net.draw_boxes(img, boxes)

    cv2.imwrite("test/pred.jpg", img)
    print(boxes.shape)
