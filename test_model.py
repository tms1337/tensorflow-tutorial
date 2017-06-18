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
from keras.datasets import cifar100

file_name = "autoencoder"

model_url = "/home/faruk/Desktop/output/%s.json" % file_name
model_file = open(model_url, "r")
model_json = model_file.read()
model_file.close()
model = model_from_json(model_json)
model.load_weights("/home/faruk/Desktop/output/%s.h5" % file_name)

(x_train, _), (x_test, _) = cifar100.load_data(label_mode='fine')

for i in range(10, 20):
    img = x_train[i]

    plt.imshow(img)
    plt.show()

    prediction = model.predict(x_train[i:i + 1])
    print(prediction.shape)
    print(prediction)
    img = prediction[0]

    plt.imshow(img)
    plt.show()
