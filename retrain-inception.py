import numpy as np

from dlm.inception_v3 import InceptionV3
from config.config import config
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.optimizers import Adam, Adadelta

print("Creating data")
batch_size = 1
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   # rotation_range=45,
                                   # width_shift_range=0.2,
                                   # height_shift_range=0.2,
                                   # zoom_range=0.5,
                                   # fill_mode="nearest",
                                   # horizontal_flip=True,
                                   # shear_range=0.2)
                                   )
data_generator = train_datagen.flow_from_directory(config["data_root_dir"],
                                                   target_size=(384, 256),
                                                   batch_size=batch_size,
                                                   shuffle=True)

total = 1000
train_perc = 0.9

print("Loading inception net")
inception = InceptionV3(include_top=False, input_shape=(384, 256, 3))

train_n = int(train_perc * total)
train_data_x = inception.predict_generator(data_generator, train_n)

class_n = 9

train_data_y = np.ndarray((train_n, class_n))
data_generator.reset()
for i in range(train_n):
    item = data_generator.next()
    train_data_y[i] = item[1]

test_n = int( (1 - train_perc) * total )
test_data_x = inception.predict_generator(data_generator, test_n)
test_data_y = np.ndarray((test_n, class_n))

for i in range(test_n):
    item = data_generator.next()
    test_data_y[i] = item[1]

model = Sequential()
model.add(Flatten(input_shape=train_data_x.shape[1:]))
model.add(Dense(8, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(class_n, activation="softmax"))

model.compile(optimizer=Adam(lr=0.00006),
              loss=binary_crossentropy,
              metrics=["accuracy"])

print("Full model compiled")

model.fit(train_data_x,
          train_data_y,
          batch_size=train_n,
          epochs=300,
          verbose=1,
          validation_data=(test_data_x, test_data_y))

print("Model fit")
