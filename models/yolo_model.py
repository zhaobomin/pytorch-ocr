import torch
from helper.utils_yolo import non_max_suppression, rescale_boxes, load_classes
import torchvision.transforms as transforms
from models import darknet_yolo
import numpy as np
import torch.nn.functional as F
from helper.dataset_ctpn import rotate_cut_img
from PIL import Image
import cv2


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size,
                          mode="nearest").squeeze(0)
    return image


class YOLO_NET:
    def __init__(self, weights=None, cfg=None, img_size=416, clss_path='weights/yolo/coco.names'):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.names = load_classes(clss_path)

        print(self.device)
        if cfg == None:
            cfg = config.yoloCfgPath
        self.img_size = img_size
        self.net = darknet_yolo.DarknetYOLO(cfg).to(self.device)

        if weights == None:
            weights = config.yoloPath

        if weights.endswith('.weights'):
            self.net.load_darknet_weights(weights)
        else:
            self.net.load_state_dict(torch.load(
                weights, map_location=self.device))
        print('loading weights:%s Done' % weights)
        self.net.eval()

    def predict(self, img, thres=0.5, nms_thres=0.3):
        with torch.no_grad():
            img = Image.fromarray(img)
            img = transforms.ToTensor()(img)
            c, h, w = img.shape
            # Pad to square resolution
            img, _ = pad_to_square(img, 0)
            # Resize
            img = resize(img, self.img_size)

            img = img.view(-1, 3, self.img_size, self.img_size)

            detections = self.net(img)  # output:[1, 22743, 7]

            detections = non_max_suppression(detections, thres, nms_thres)[0]
            if detections == None:
                return None

            detections = rescale_boxes(detections, self.img_size, (h, w))
            output = []
            for res in detections:
                x1, y1, x2, y2, conf, cls_conf, cls = res
                clsid = int(cls.item())
                clsname = self.names[clsid]
                if conf < thres:
                    continue
                output.append([int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item()),
                               conf.item(), cls_conf.item(), clsid, clsname])

            return output

    def draw_boxes(self, img, boxes):
        for b in boxes:
            x1, y1, x2, y2, conf, cls_conf, clsid, clsname = b
            img = cv2.rectangle(img, (x1, y1),
                                (x2, y2), (255, 0, 0), 2)
            img = cv2.rectangle(img, (x1, y1-10),
                                (x2, y1), (255, 0, 0), 2)
            cv2.putText(img, "%s %.2f%%" % (clsname, conf*100), (x1, y1), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 0), thickness=2)
        return img

    def rotate_cut_img(self, im, box, leftAdjust=0.0, rightAdjust=0.0):
        return rotate_cut_img(im, box, leftAdjust, rightAdjust)
