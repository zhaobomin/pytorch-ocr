import config
from model_crnn import crnn_model
import time
from PIL import Image
import sys


if __name__ == '__main__':
    t = time.time()
    img = Image.open(sys.argv[1])
    net = crnn_model.crnn_net(
        config.ocrPath, config.ocrCfgPath, config.charactersPath)
    res = net.predict(img)
    print(time.time()-t, res)
