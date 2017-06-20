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
from keras.models import load_model
from keras.datasets import cifar100
from ConvolutionalAutoencoderLayer import ConvolutionalAutoencoderLayer

file_name = "conv-autoencoder"
model = load_model("/home/faruk/Desktop/output/%s.h5" % file_name,
                   custom_objects={"ConvolutionalAutoencoderLayer": ConvolutionalAutoencoderLayer})

(x_train, _), (x_test, _) = cifar100.load_data(label_mode='fine')

for i in range(0, 100):
    img = x_train[i]

    plt.imshow(img)
    plt.show()

    prediction = model.predict(x_train[i:i + 1])
    img = prediction[0]

    plt.imshow(img)
    plt.show()
