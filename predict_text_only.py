import sys
import cv2
from models import text_model
if __name__ == '__main__':

    img = cv2.imread(sys.argv[1])
    net = text_model.TEXT_NET('weights/text/text_training.pth')

    boxes, scores = net.predict(img)
	#keep = scores > 0.7
    print(scores)
    img = net.draw_boxes(img, boxes[scores > 0.7])

    cv2.imwrite("test/pred.jpg", img)
    print(boxes.shape)
