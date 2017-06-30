import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.losses import mean_squared_error

file_name = "HIGGS"
nrows = int(1e4)

print("Loading data with %d rows " % nrows)
df = pd.read_csv("/home/faruk/workspace/thesis/data/%s.dat" % file_name,
                 header=None,
                 nrows=nrows,
                 chunksize=nrows,
                 index_col=False)
df = df.read(nrows)
df_norm = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df.values))

data_x = df_norm.iloc[:, 1:].as_matrix()
data_y = df.iloc[:, 0].as_matrix()

y_before = data_y

should_permute = True
if should_permute:
    noise_lvl = 0.2
    noise_n = int(noise_lvl * data_y.shape[0])

    permutation = 1 - data_y[:noise_n]
    data_y = np.concatenate( (permutation, data_y[noise_n:]) )

print(data_x.shape, data_y.shape)
data = np.column_stack( (data_x, data_y) )

filter_factor = 0.9
filter_n = int(data.shape[1] * filter_factor)
denoiser = Sequential()
denoiser.add(Dense(filter_n, input_shape=data.shape[1:], activation="relu"))
denoiser.add(Dropout(0.2))
denoiser.add(Dense(data.shape[1], activation="relu"))

denoiser.compile(optimizer=Adam(lr=0.0003),
                 loss=mean_squared_error,
                 metrics=[mean_squared_error])
denoiser.fit(data, data,
             batch_size=250,
             epochs=1000,
             verbose=1)

changed_x = data[:, :-1]
changed_y = data[:, -1]
changed_y = 1 - changed_y
changed_data = np.column_stack( (changed_x, changed_y) )

denoiser.compile(optimizer=Adam(lr=0.00001),
                 loss=mean_squared_error,
                 metrics=[mean_squared_error])
denoiser.fit(changed_data, data,
             batch_size=250,
             epochs=1000,
             verbose=1)

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
