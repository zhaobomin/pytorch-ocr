from config import ocrPath
from models import crnn_model
import time
from PIL import Image
import sys


if __name__ == '__main__':
    t = time.time()
    img = Image.open(sys.argv[1])
    net = crnn_model.CRNN_NET(ocrPath)
    res = net.predict(img)
    print(time.time()-t, res)
