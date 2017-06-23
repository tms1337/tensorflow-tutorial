from ConvolutionalAutoencoderLayer import ConvolutionalAutoencoderLayer
from keras.datasets import cifar100
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import mean_squared_error, binary_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from config.config import config
from ConvolutionalAutoencoderLayer import ConvolutionalAutoencoderLayer
from keras.callbacks import ModelCheckpoint

(x_train, _), (x_test, _) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype("float32") * (1.0 / 255)
x_test = x_test.astype("float32") * (1.0 / 255)

data_n = config["conv-autoencoder"]["data_n"]
if data_n is not None:
    x_train = x_train[0:data_n]
    x_test = x_test[0:data_n]

epoch_n = config["conv-autoencoder"]["epoch_n"]
batch_size = config["conv-autoencoder"]["batch_size"]

input_shape = (32, 32, 3)

autoencoder = Sequential()
autoencoder.add(ConvolutionalAutoencoderLayer(512, filter_size=(3, 3), input_shape=input_shape))

autoencoder.compile(optimizer=Adam(lr=0.0001),
                    loss=binary_crossentropy,
                    metrics=[binary_crossentropy])

checkpointer = ModelCheckpoint(filepath="%s/%s" % (config["output_dir"], "conv-autoencoder.h5"),
                               verbose=1,
                               period=5,
                               save_weights_only=True,
                               save_best_only=False)

autoencoder.fit(x_train, x_train,
                batch_size=batch_size,
                epochs=epoch_n,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[checkpointer])
