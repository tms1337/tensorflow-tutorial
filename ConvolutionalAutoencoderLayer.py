import numpy as np
from keras import backend as K
from keras.engine import Layer


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
        self.encoder_bias = self.add_weight(name="encoder_bias",
                                            shape=(self.filters,),
                                            initializer="uniform",
                                            trainable=True)

        self.decoder_bias = self.add_weight(name="decoder_bias",
                                            shape=(input_shape[3],),
                                            initializer="uniform",
                                            trainable=True)

        super(ConvolutionalAutoencoderLayer, self).build(input_shape)

    # noinspection PyMethodOverriding
    def call(self, x):
        h = K.conv2d(x, self.kernel, padding="same", data_format="channels_last")
        h = K.bias_add(h, self.encoder_bias)
        h = K.relu(h)

        W = self.get_weights()[0]
        transpose_shape = list(range(len(W.shape)))
        transpose_shape[-2], transpose_shape[-1] = transpose_shape[-1], transpose_shape[-2]
        W_transpose = np.transpose(W, transpose_shape)

        y = K.conv2d(h, W_transpose, padding="same", data_format="channels_last")
        y = K.bias_add(y, self.decoder_bias)
        y = K.relu(y)

        print(y.shape)

        return y

    def compute_output_shape(self, input_shape):
        return (None, input_shape)

    def get_encoder_weights(self):
        return self.get_weights()[0], self.get_weights()[1]

    def get_config(self):
        return {"filters": self.filters,
                "filter_size": self.filter_size,
                "input_shape": self.input_shape}

