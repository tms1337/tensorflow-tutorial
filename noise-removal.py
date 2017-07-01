import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, Adamax
from keras.losses import mean_squared_error, categorical_crossentropy

from clrcallback import CyclicLR
from plotloss import PlotLosses

from config.config import config

file_name = "HIGGS"
nrows = int(1e3)

print("Loading data with %d rows " % nrows)
file = config["noise-removal"]["input_file"]

if config["noise-removal"]["is_compressed"]:
    df = pd.read_csv(file,
                     compression="gzip",
                     header=None,
                     nrows=nrows,
                     chunksize=nrows,
                     index_col=False)
else:
    df = pd.read_csv(file,
                     header=None,
                     nrows=nrows,
                     chunksize=nrows,
                     index_col=False)

df = df.read(nrows)
df_norm = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df.values))

data_x = df_norm.iloc[:, 1:].as_matrix()
data_y = df.iloc[:, 0].as_matrix()

y_before_noise = data_y

should_permute = True
if should_permute:
    noise_lvl = 0.2
    noise_n = int(noise_lvl * data_y.shape[0])

    permutation = 1 - data_y[:noise_n]
    data_y = np.concatenate( (permutation, data_y[noise_n:]) )

print("Before: %d of %d" % (np.count_nonzero(data_y == y_before_noise), data_y.shape[0]) )

coding_factor = 0.8
code_length = int(coding_factor * data_x.shape[1])

print(data_x.shape)

autoencoder = Sequential()
# autoencoder.add(Dense(16*data_x.shape[1], input_shape=data_x.shape[1:], activation="relu"))
autoencoder.add(Dense(code_length, input_shape=data_x.shape[1:], activation="relu"))
# autoencoder.add(Dense(16*data_x.shape[1], activation="relu"))
# autoencoder.add(Dropout(0.5))
autoencoder.add(Dense(data_x.shape[1], activation="softsign"))

def min_square_diff(y_true, y_pred):
    return np.min(np.square(y_true - y_pred))

autoencoder.compile(optimizer=Adamax(),
                    loss=min_square_diff,
                    metrics=[min_square_diff])

autoencoder_cyc = CyclicLR(base_lr=0.001,
                                max_lr=0.003,
                                mode="triangular2")

if config["noise-removal"]["plot"]:
    callbacks = [autoencoder_cyc]
else:
    callbacks = [autoencoder_cyc]

autoencoder.fit(data_x, data_x,
                batch_size=250,
                epochs=1250,
                callbacks=callbacks)

encoder = Sequential()
# encoder.add(Dense(16*data_x.shape[1], input_shape=data_x.shape[1:], activation="relu", weights=autoencoder.layers[0].get_weights()))
encoder.add(Dense(code_length, input_shape=data_x.shape[1:], activation="relu", weights=autoencoder.layers[0].get_weights()))

encoded_x = encoder.predict(data_x)

targets = data_y.reshape(-1)
one_hot_targets = np.eye(2)[targets.astype(int)]

batch_size = 250

classifier = Sequential()
# classifier.add(BatchNormalization(batch_input_shape=(None, encoded_x.shape[1]), batch_size=batch_size))
classifier.add(Dense(encoded_x.shape[1]*16, input_shape=encoded_x.shape[1:], activation="relu"))
classifier.add(Dense(one_hot_targets.shape[1], activation="softmax"))

classifier.compile(optimizer=Adamax(),
                   loss=categorical_crossentropy,
                   metrics=[categorical_crossentropy])

classifier_cyclic_lr = CyclicLR(base_lr=0.001,
                                max_lr=0.003,
                                mode="triangular2")

if config["noise-removal"]["plot"]:
    callbacks = [classifier_cyclic_lr, PlotLosses()]
else:
    callbacks = [classifier_cyclic_lr]

classifier.fit(encoded_x, one_hot_targets,
               batch_size=batch_size,
               epochs=700,
               callbacks=callbacks)

predictions = np.argmax(np.round(classifier.predict(encoded_x)), axis=1)

print("After: %d of %d" % (np.count_nonzero(predictions == y_before_noise), data_y.shape[0]) )
