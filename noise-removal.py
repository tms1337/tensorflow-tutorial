import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, Adamax
from keras.losses import mean_squared_error, categorical_crossentropy

from clrcallback import CyclicLR
from plotloss import PlotLosses

from config.config import config

file_name = "HIGGS"
nrows = int(5e5)

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

targets = data_y.reshape(-1)
one_hot_targets = np.eye(2)[targets.astype(int)]

batch_size = 500

classifier = Sequential()
classifier.add(BatchNormalization(batch_input_shape=(None, data_x.shape[1]), batch_size=batch_size))
classifier.add(Dense(data_x.shape[1]*16, input_shape=data_x.shape[1:], activation="relu"))
classifier.add(Dense(data_x.shape[1]*16, activation="relu"))
classifier.add(Dense(one_hot_targets.shape[1], activation="softmax"))

classifier.compile(optimizer=Adamax(),
                   loss=categorical_crossentropy,
                   metrics=[categorical_crossentropy])

classifier_cyclic_lr = CyclicLR(base_lr=0.01,
                                max_lr=0.03,
                                mode="triangular2")

if config["noise-removal"]["plot"]:
    callbacks = [classifier_cyclic_lr, PlotLosses()]
else:
    callbacks = [classifier_cyclic_lr]

classifier.fit(data_x, one_hot_targets,
               batch_size=batch_size,
               epochs=2000,
               callbacks=callbacks)


y_before = data_y

should_permute = True
if should_permute:
    noise_lvl = 0.2
    noise_n = int(noise_lvl * data_y.shape[0])

    permutation = 1 - data_y[:noise_n]
    data_y = np.concatenate( (permutation, data_y[noise_n:]) )

print(data_x.shape, data_y.shape)
data = np.column_stack( (data_x, data_y) )

filter_factor = 0.5
filter_n = int(data.shape[1] * filter_factor)
denoiser = Sequential()
denoiser.add(Dense(8*data.shape[1], input_shape=data.shape[1:], activation="relu"))
denoiser.add(Dropout(0.5))
denoiser.add(Dense(filter_n, activation="relu"))
denoiser.add(Dense(8*data.shape[1], activation="relu"))
denoiser.add(Dropout(0.5))
denoiser.add(Dense(data.shape[1], activation="softsign"))

denoiser.compile(optimizer=Adamax(),
                 loss=mean_squared_error,
                 metrics=[mean_squared_error])

cyclic_lr = CyclicLR(base_lr=0.0001, max_lr=0.001)

changed_x = np.copy(data[:, :-1])
changed_y = np.copy(data[:, -1])
changed_y = 1 - changed_y
changed_data = np.column_stack( (changed_x, changed_y) )

denoiser.compile(optimizer=Adam(),
                 loss=mean_squared_error,
                 metrics=[mean_squared_error])

if config["noise-removal"]["plot"]:
    callbacks = [cyclic_lr, PlotLosses()]
else:
    callbacks = [cyclic_lr]

denoiser.fit(changed_data, data,
             batch_size=500,
             epochs=1000,
             verbose=1,
             callbacks=callbacks)

knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
score = cross_val_score(knn,
                        data_x,
                        data_y,
                        cv=10,
                        n_jobs=-1,
                        verbose=0)
avg_score = np.average(np.array(score))
print("Before %f" % avg_score)

data = denoiser.predict(data)

print( "Count before: ", np.count_nonzero(data_y == y_before) )

# data_x = data[:,:-1]
data_y = np.round( data[:,-1] )

print("Size: ", data_y.shape[0])
print( "Count after: ", np.count_nonzero(data_y == y_before) )

score = cross_val_score(knn,
                        data_x,
                        data_y,
                        cv=10,
                        n_jobs=-1,
                        verbose=0)
avg_score = np.average(np.array(score))
print("After %f" % avg_score)
