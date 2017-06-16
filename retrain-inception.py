import numpy as np

from dlm.inception_v3 import InceptionV3
from config.config import config
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, Sequential, model_from_json
from keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error
from keras.optimizers import Adam, Adadelta
import keras.backend as K

print("Creating data")
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
                                                   target_size=(384, 256),
                                                   batch_size=batch_size,
                                                   shuffle=True)

print("Loading inception net")
inception_model_path = "%s/%s/%s" % (config["data_root_dir"], "inception_model", "inception.json")
inception_file = open(inception_model_path, "r")
inception_json = inception_file.read()
inception_file.close()
inception = model_from_json(inception_json)
inception_weights_path = "%s/%s/%s" % (config["data_root_dir"], "inception_model", "inception.h5")
inception.load_weights(inception_weights_path)

code_length = 1024

model = Sequential()
model.add(Flatten(input_shape=(10, 6, 2048)))
model.add(Dropout(0.5))
model.add(Dense(code_length, activation="softmax"))

model.compile(optimizer=Adam(lr=0.0003),
              loss=mean_squared_error,
              metrics=[mean_squared_error])

data_n = config["inception-top"]["data_n"]
train_x = np.ndarray((data_n, 10, 6, 2048))
train_y = np.ndarray((data_n, code_length))
for i in range(data_n):
    a = data_generator.next()
    b = data_generator.next()

    train_x[i] = inception.predict(a[0])

    y_intermediate = inception.predict(b[0])
    y = np.round(model.predict(y_intermediate), 0)
    if not np.array_equal(a[1], b[1]):
        y = 1 - y

    train_y[i] = y

batch_size = config["inception-top"]["batch_size"]
epoch_n = config["inception-top"]["epoch_n"]
model.fit(train_x, train_y, batch_size=batch_size, epochs=epoch_n)
