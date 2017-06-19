import numpy as np

from dlm.inception_v3 import InceptionV3
from config.config import config
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from keras.layers import Flatten, Dense, Dropout, MaxPooling2D, Conv2D, UpSampling2D, Lambda
from keras.models import Model, Sequential, model_from_json
from keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error
from keras.optimizers import Adam, Adadelta
import keras.backend as K
from keras.datasets import cifar100
import sys

# print("Creating data")
# width = 256
# height = 256

# batch_size = 1
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   # width_shift_range=0.2,
                                   # height_shift_range=0.2,
                                   # zoom_range=0.5,
                                   # fill_mode="nearest",
                                   # horizontal_flip=True,
                                   # shear_range=0.2)
                                   )
# data_path = "%s/%s" % (config["data_root_dir"], "wang999")
# data_generator = train_datagen.flow_from_directory(data_path,
#                                                    target_size=(width, height),
#                                                    batch_size=batch_size,
#                                                    shuffle=True)

(x_train, _), (x_test, _) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype("float32") * (1.0 / 255)
x_test = x_test.astype("float32") * (1.0 / 255)

code_length = 128

autoencoder = Sequential()

autoencoder.add(Conv2D(16, (2, 2), activation="relu", padding="same", input_shape=x_train.shape[1:]))
autoencoder.add(MaxPooling2D((2, 2)))

autoencoder.add(Dropout(0.5))

autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (2, 2), activation="relu", padding="same"))

autoencoder.add(Conv2D(3, (32, 32), activation="softmax", padding="same"))

autoencoder.compile(optimizer=Adam(lr=0.001),
                    loss=mean_squared_error,
                    metrics=[mean_squared_error])

epoch_n = config["inception-top"]["epoch_n"]
batch_size = config["inception-top"]["batch_size"]
steps_per_epoch = config["inception-top"]["steps_per_epoch"]

# data_generator = train_datagen.flow(x_train, x_train, shuffle=True, batch_size=batch_size)

# autoencoder.fit_generator(data_generator, steps_per_epoch=steps_per_epoch, epochs=epoch_n,
#                           validation_data=(x_test, x_test))

autoencoder.fit(x_train, x_train,
                batch_size=batch_size,
                epochs=epoch_n,
                shuffle=True,
                validation_data=(x_test, x_test))

model_json = autoencoder.to_json()
with open("%s/%s" % (config["output_dir"], "autoencoder.json"), "w+") as model_file:
    model_file.write(model_json)
autoencoder.save_weights("%s/%s" % (config["output_dir"], "autoencoder.h5"))
