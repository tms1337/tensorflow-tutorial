import pandas as pd
import numpy as np
import io
import requests
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, GaussianNoise, Input
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, mean_squared_error
from keras.activations import softsign, relu
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

file_name = "HIGGS"
nrows = int(2e6)

print("Loading data with %d rows " % nrows)
df = pd.read_csv("/home/faruk/workspace/thesis/data/%s.dat" % file_name, header=None, nrows=nrows, chunksize=nrows, index_col=False)
df = df.read(nrows)
df_norm = pd.DataFrame( preprocessing.StandardScaler().fit_transform(df.values) )
train_x = df_norm.iloc[:,1:].as_matrix()
train_y = df.iloc[:, 0]

noise_level = 0.2

rff = RandomForestClassifier(n_estimators=10, n_jobs=4)
rff.fit(train_x[:-int(nrows/10)], train_y[:-int(nrows/10)])

test_x = train_x[:int(nrows / 10):]
noise_level_per_col = noise_level * np.std(test_x, 0)
test_x += np.random.normal(0, noise_level_per_col, test_x.shape)

correct = np.count_nonzero(rff.predict(test_x) == train_y[:int(nrows / 10):])
print("Initial accuracy: ", (1.0 * correct)/(nrows/10))

col_n = len(df.columns) - 1
input_shape = (col_n,)

print("Constructing model")

denoiser = Sequential()
denoiser.add(GaussianNoise(noise_level, input_shape=input_shape))
denoiser.add(Dense(int(8 * col_n), activation=relu))
denoiser.add(Dense(int(col_n), activation=relu))
denoiser.add(Dropout(0.1))
denoiser.add(Dense(int(8 * col_n), activation=relu))
denoiser.add(Dense(col_n, activation=softsign))

denoiser.compile(optimizer=Adam(lr=0.0003, decay=0.00001),
                 loss=mean_squared_error,
                 metrics=[mean_squared_error])

early_stopping_callback = EarlyStopping(patience=5,
                                        min_delta=0.0001)
denoiser.fit(train_x, train_x,
             batch_size=250,
             epochs=100,
             shuffle=True,
             validation_split=0.1,
             callbacks=[early_stopping_callback],
             verbose=1)

noise_level_per_col = noise_level * np.std(train_x, 0)
train_x += np.random.normal(0, noise_level_per_col, train_x.shape)
out_df = pd.DataFrame(data=denoiser.predict(train_x))
out_df[len(out_df.columns)] = train_y
out_df.to_csv("/home/faruk/Desktop/%s-pca.csv" % file_name, index=False, header=False)

rff = RandomForestClassifier(n_estimators=20, n_jobs=4)
validation_n = int(nrows / 10)
rff.fit(out_df.iloc[:-validation_n, :-1].as_matrix(), out_df.iloc[:-validation_n, -1].as_matrix())

test_x = out_df.iloc[-validation_n:, :-1].as_matrix()
test_y = out_df.iloc[-validation_n:, -1].as_matrix()

correct = 1.0 * np.count_nonzero(rff.predict(test_x) == test_y)
print(correct / test_y.shape[0])

exit(0)

plt.gray()
for i in range(100):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    img = np.absolute(train_x[i].reshape(3, 6) - denoiser.predict(train_x[i].reshape(1, 18)).reshape(3, 6))
    imgplot = plt.imshow(img)
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7, 0.9], orientation='horizontal')
    a.set_title('Diff')

    plt.show()
