from config.config import config

import keras
import keras.backend as K

from keras.models import Sequential

from keras.layers import Layer
from keras.datasets import cifar100
from keras.optimizers import Adam
from keras.losses import mean_squared_error

import numpy as np

class ConvolutionalAutoencoderLayer(Layer):
    def __init__(self,
                 filters,
                 filter_size=(2, 2),
                 **kwargs):
        """
        Constructor for creating the ConvAutoencoder Layer
        Inherited from the keras.layers.Layer base class

        :param filters: number of filters to be applied
        :param filter_size: size of the filter in (width, height) format
        """

        self.filters = filters
        self.filter_size = filter_size

        super(ConvolutionalAutoencoderLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (self.filter_size[0], self.filter_size[1], input_shape[3], self.filters)
        self.kernel = self.add_weight(name="kernel",
                                      shape=kernel_shape,
                                      initializer="uniform",
                                      trainable=True)

        print(self.kernel)

        super(ConvolutionalAutoencoderLayer, self).build(input_shape)

    # noinspection PyMethodOverriding
    def call(self, x):
        h = K.conv2d(x, self.kernel, padding="same", data_format="channels_last")

        W = self.get_weights()[0]
        transpose_shape = range(len(W.shape))
        transpose_shape[-2], transpose_shape[-1] = transpose_shape[-1], transpose_shape[-2]
        W_transpose = np.transpose(W, transpose_shape)

        y = K.conv2d(h, W_transpose, padding="same", data_format="channels_last")

        return y

    def compute_output_shape(self, input_shape):
        return input_shape


(x_train, _), (x_test, _) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype("float32") * (1.0 / 255)
x_test = x_test.astype("float32") * (1.0 / 255)

epoch_n = config["deep-conv-autoencoder"]["epoch_n"]
batch_size = config["deep-conv-autoencoder"]["batch_size"]

model = Sequential()

layer_n = 2
for i in range(layer_n):
    if i == 0:
        input_shape = (32, 32, 3)
        model.add(ConvolutionalAutoencoderLayer(64, filter_size=(2, 2), input_shape=input_shape))
    else:
        model.add(ConvolutionalAutoencoderLayer(64, filter_size=(2, 2)))

    model.compile(optimizer=Adam(lr=0.001),
                  loss=mean_squared_error,
                  metrics=[mean_squared_error])

    model.fit(x_train[0:4], x_train[0:4],
              batch_size=batch_size,
              epochs=epoch_n,
              shuffle=True,
              validation_data=(x_test[0:4], x_test[0:4]))

model_json = model.to_json()
with open("%s/%s" % (config["output_dir"], "conv-autoencoder.json"), "w+") as model_file:
    model_file.write(model_json)
model.save_weights("%s/%s" % (config["output_dir"], "conv-autoencoder.h5"))
