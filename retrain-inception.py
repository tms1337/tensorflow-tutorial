from dlm.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

inception = InceptionV3(include_top=False, input_shape=(256, 384, 3))
# x = Flatten()(inception.layers[-1].output)
# x = Dense(1024, activation="relu")(x)
# last_layer = Dense(10, activation="softmax")(x)

model = Sequential()
model.add(inception.layers[-1])

for layer in model.layers:
    layer.trainable = False

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer=Adam(),
              loss=binary_crossentropy,
              metrics=["accuracy"])


