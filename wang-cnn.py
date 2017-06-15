from __future__ import division, print_function

import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D

import pylab
from PIL import Image
import os
import random
import matplotlib.pyplot as plt


is_local = False
model_type = "cnn"

if is_local:
    train_perc = 0.1
else:
    train_perc = 0.9

if is_local:
    root_url = "/home/faruk/Desktop/wang1000"
else:
    root_url = "/input"

folders = [str(i) for i in range(1, 10)]
total = 1000

x_train = np.ndarray((total, 256, 256, 3))
y_train = np.ndarray((total, 1))
train_n = 0

x_test = np.ndarray((total, 256, 256, 3))
y_test = np.ndarray((total, 1))
test_n = 0

for folder in folders:
    for file in os.listdir("%s/%s" % (root_url, folder)):
        full_file_path = "%s/%s/%s" % (root_url, folder, file)
        img = Image.open(open(full_file_path, "rb"))
        img = np.asarray(img, dtype='float64') / 256.

        img_tensor = img.transpose(2, 0, 1).reshape(1, 384, 256, 3)
        img_tensor = img_tensor[:, 0:256, :, :]

        if not is_local or random.random() < 0.1:
            if random.random() < train_perc:
                x_train[train_n] = img_tensor
                y_train[train_n] = folder
                train_n += 1
            else:
                x_test[test_n] = img_tensor
                y_test[test_n] = folder
                test_n += 1

x_train = x_train[0:train_n]
y_train = y_train[0:train_n]

x_test = x_test[0:test_n]
y_test = y_test[0:test_n]

print(x_train.shape)
print(x_test.shape)

model = Sequential()

if model_type == "flat":
    encoding_dim = 100
    model.add(Dense(256, activation="relu", input_shape=(384 * 256 * 3,)))
    model.add(Dense(encoding_dim, activation="relu"))
    model.add(Dense(256, activation="relu", input_shape=(384 * 256 * 3,)))
    model.add(Dense(384 * 256 * 3, activation="sigmoid"))
    x_train = x_train.reshape((train_n, 384 * 256 * 3))
    x_test = x_test.reshape((test_n, 384 * 256 * 3))
elif model_type == "cnn":
    model.add(MaxPooling2D((2, 2), padding="same", input_shape=(256, 256, 3)))
    model.add(Conv2D(64, (2, 2), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), padding="same"))
    model.add(Conv2D(32, (4, 4), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Conv2D(32, (4, 4), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (2, 2), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (2, 2), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam())

if is_local:
    batch_size = 40
else:
    batch_size = 100

if is_local:
    epochs = 2
else:
    epochs = 500

model.fit(x_train, x_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True)

def write_model_to_file(model, name):
    if is_local:
        output_dir = "/home/faruk/Desktop"
    else:
        output_dir = "/output"

    model_file_name = "%s/%s.json" % (output_dir, name)
    model_json = model.to_json()
    with open(model_file_name, "w+") as model_file:
        model_file.write(model_json)

    weights_file_name = "%s/%s.h5" % (output_dir, name)
    model.save_weights(weights_file_name)

write_model_to_file(model, "full")
