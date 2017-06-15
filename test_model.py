from __future__ import division, print_function

import numpy as np

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

import pylab
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

model_url = "/home/faruk/Desktop/output/full.json"
model_file = open(model_url, "r")
model_json = model_file.read()
model_file.close()
model = model_from_json(model_json)
model.load_weights("/home/faruk/Desktop/output/full.h5")

img = Image.open(open("/home/faruk/Desktop/wang1000/1/132.jpg", "rb"))
img = np.asarray(img, dtype='float64') / 256.

plt.imshow(img)
plt.show()

img_input = img.reshape((1, 384*256*3))
decoded_img_tensor = 256.0 * model.predict(img_input)

decoded_img = decoded_img_tensor.reshape( (384, 256, 3) )
plt.imshow(decoded_img)
plt.show()
