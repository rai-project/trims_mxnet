import os, sys
from skimage import io, transform
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../../../python"))
import mxnet as mx
from mxnet_predict import Predictor, load_ndarray_file


ctx = mx.gpu(0)

path='http://data.mxnet.io/models/imagenet-11k/'
[mx.test_utils.download(path+'resnet-152/resnet-152-symbol.json'),
 mx.test_utils.download(path+'resnet-152/resnet-152-0000.params'),
 mx.test_utils.download(path+'synset.txt')]

# Load the pre-trained model
prefix = "resnet-152"
num_round = 0
symbol_file = "%s-symbol.json" % prefix
param_file = "%s-0000.params" % prefix
predictor = Predictor(open(symbol_file, "r").read(),
                      open(param_file, "rb").read(),
                                            {'data':(1, 3, 224, 224)})

synset = [l.strip() for l in open('synset.txt').readlines()]

import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(url):
    # download and show the image
    fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def PreprocessImage(path):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 255
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    # sub mean
    return sample


def predict(url):
    img = get_image(url)

    predictor.forward(data=img)
    prob = predictor.get_output(0)[0]

    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1)
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print("Top5: ", top5)


predict('http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg')
predict('http://thenotoriouspug.com/wp-content/uploads/2015/01/Pug-Cookie-1920x1080-1024x576.jpg')
