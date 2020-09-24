from model_crnn import crnn_model
import time
from PIL import Image
import sys
import config

if __name__ == '__main__':

    net = crnn_model.crnn_net(
        config.ocrPath, config.ocrCfgPath, config.charactersPath)
    t = time.time()
    img = Image.open(sys.argv[1])

    res = net.predict(img)
    print(time.time()-t, res)
