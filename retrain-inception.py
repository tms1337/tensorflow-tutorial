import numpy as np

from dlm.inception_v3 import InceptionV3
from config.config import config
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from keras.layers import Flatten, Dense, Dropout, MaxPooling2D, Conv2D, UpSampling2D, Lambda
from keras.models import Model, Sequential, model_from_json
from keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error
from keras.optimizers import Adam, Adadelta
import keras.backend as K
import sys

print("Creating data")
width = 256
height = 256

batch_size = 1
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.5,
                                   fill_mode="nearest",
                                   horizontal_flip=True,
                                   shear_range=0.2)
data_path = "%s/%s" % (config["data_root_dir"], "wang999")
data_generator = train_datagen.flow_from_directory(data_path,
                                                   target_size=(width, height),
                                                   batch_size=batch_size,
                                                   shuffle=True)

autoencoder = Sequential()

autoencoder.add(Conv2D(32, (2, 2), activation="relu", padding="same", input_shape=(width, height, 3)))
autoencoder.add(MaxPooling2D((4, 4)))

autoencoder.add(Conv2D(32, (2, 2), activation="relu", padding="same"))
autoencoder.add(MaxPooling2D((4, 4)))

autoencoder.add(Conv2D(16, (2, 2), activation="relu", padding="same"))
autoencoder.add(MaxPooling2D((4, 4)))

autoencoder.add(Conv2D(16, (2, 2), activation="relu", padding="same"))
autoencoder.add(MaxPooling2D((4, 4)))

autoencoder.add(Dropout(0.5))

autoencoder.add(Conv2D(32, (1, 1), activation="softmax", padding="same"))
autoencoder.add(UpSampling2D((4, 4)))

autoencoder.add(Conv2D(16, (2, 2), activation="relu", padding="same"))
autoencoder.add(UpSampling2D((4, 4)))

autoencoder.add(Conv2D(16, (2, 2), activation="relu", padding="same"))
autoencoder.add(UpSampling2D((4, 4)))

autoencoder.add(Conv2D(16, (2, 2), activation="relu", padding="same"))
autoencoder.add(UpSampling2D((4, 4)))

autoencoder.add(Dropout(0.5))

autoencoder.add(Conv2D(3, (4, 4), activation="softmax", padding="same"))

autoencoder.compile(optimizer=Adam(lr=0.001),
                    loss=binary_crossentropy,
                    metrics=[binary_crossentropy])

total = config["inception-top"]["data_n"]
data = np.ndarray((total, width, height, 3))
for i in range(total):
    data[i] = data_generator.next()[0]

print(data)
print(data.shape)

epoch_n = config["inception-top"]["epoch_n"]
batch_size = config["inception-top"]["batch_size"]
autoencoder.fit(data, data, batch_size=batch_size, epochs=epoch_n)
