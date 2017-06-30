from sklearn.neural_network import BernoulliRBM

import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

file_name = "HIGGS"
nrows = int(1e5)

print("Loading data with %d rows " % nrows)
df = pd.read_csv("/home/faruk/workspace/thesis/data/%s.dat" % file_name,
                 header=None,
                 nrows=nrows,
                 chunksize=nrows,
                 index_col=False)

df = df.read(nrows)
df_norm = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df.values))
data_x = df_norm.iloc[:, 1:].as_matrix()
data_y = df.iloc[:, 0]

train_perc = 0.1
train_n = int(data_x.shape[0] * train_perc)

train_x = data_x[0:-train_n]
test_x = data_x[-train_n:]

noise_level = 0.2

train_x_noise = train_x
noise_level_per_col = noise_level * np.std(train_x, 0)
train_x_noise += np.random.normal(0, noise_level_per_col, train_x.shape)

factor = 0.75
rbm = BernoulliRBM(learning_rate=0.0001,
                   batch_size=500,
                   n_components=int(data_x.shape[1] * factor),
                   n_iter=100,
                   verbose=2)

rbm.fit(train_x_noise, train_x)
print(rbm.get_params())
