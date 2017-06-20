from keras.datasets import cifar100
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.optimizers import Adam

from ConvolutionalAutoencoderLayer import ConvolutionalAutoencoderLayer
from config.config import config

(x_train, _), (x_test, _) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype("float32") * (1.0 / 255)
x_test = x_test.astype("float32") * (1.0 / 255)

epoch_n = config["deep-conv-autoencoder"]["epoch_n"]
batch_size = config["deep-conv-autoencoder"]["batch_size"]

input_shape = (32, 32, 3)

model = Sequential()

layer_n = 1
autoencoders = []
for i in range(layer_n):
    autoencoder = Sequential()

    if i == 0:
        autoencoder.add(ConvolutionalAutoencoderLayer(64, filter_size=(2, 2), input_shape=input_shape))
    else:
        autoencoder.add(ConvolutionalAutoencoderLayer(64, filter_size=(2, 2), input_shape=model.output_shape[1:]))

    autoencoder.compile(optimizer=Adam(lr=0.001),
                        loss=mean_squared_error,
                        metrics=[mean_squared_error])

    if i == 0:
        x = x_train
        y = x_test
    else:
        x = model.predict(x_train)
        y = model.predict(x_test)

    autoencoder.fit(x, x,
                    batch_size=batch_size,
                    epochs=epoch_n*(i + 1),
                    shuffle=True,
                    validation_data=(y, y))

    autoencoders.append(autoencoder)

    weights = autoencoder.layers[0].get_encoder_weights()
    if i == 0:
        model.add(Conv2D(64, (2, 2), activation="relu", padding="same", weights=weights, input_shape=input_shape))
    else:
        model.add(Conv2D(64, (2, 2), activation="relu", padding="same", weights=weights))

    model.add(MaxPooling2D((2, 2)))
    model.compile(optimizer=Adam(lr=0.001),
                  loss=mean_squared_error,
                  metrics=[mean_squared_error])

model_json = model.to_json()
with open("%s/%s" % (config["output_dir"], "deep-conv-autoencoder.json"), "w+") as model_file:
    model_file.write(model_json)
model.save_weights("%s/%s" % (config["output_dir"], "deep-conv-autoencoder.h5"))
