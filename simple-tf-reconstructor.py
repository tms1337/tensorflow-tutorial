import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

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
data_y = df.iloc[:, 0]

noise_level = 0.2

rff = RandomForestClassifier(n_estimators=10, n_jobs=4)
rff.fit(data_x[:-int(nrows / 10)], data_y[:-int(nrows / 10)])

test_x = data_x[-int(nrows / 10):]
test_x_noisy = test_x
noise_level_per_col = noise_level * np.std(test_x_noisy, 0)
test_x_noisy += np.random.normal(0, noise_level_per_col, test_x_noisy.shape)
test_y = data_y[-int(nrows / 10):]

correct = np.count_nonzero(rff.predict(test_x_noisy) == test_y)
print("Initial accuracy (noisy data): ", (1.0 * correct) / (nrows / 10))

class BatchGenerator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.cnt = -1

    def get_batch(self):
        self.cnt += 1
        return self.data[self.cnt*self.batch_size : (self.cnt+1)*self.batch_size]


def denoiser(input_n, hidden_n, noise_lvl):
    x = tf.placeholder(tf.float32, shape=[None, input_n], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, input_n], name="y_")

    noise = tf.random_normal(shape=[input_n, ], mean=0.0, stddev=noise_lvl, dtype=tf.float32)
    x_noise = x + noise

    W = tf.Variable(tf.random_normal([input_n, hidden_n], stddev=0.5))
    b_visible = tf.Variable(tf.random_normal([hidden_n, ], stddev=0.5))
    b_hidden = tf.Variable(tf.random_normal([input_n, ], stddev=0.5))

    h = tf.nn.sigmoid( tf.matmul(x_noise, W) + b_visible )
    y = tf.nn.softsign( tf.matmul(h, tf.transpose(W)) + b_hidden )

    loss = tf.reduce_mean( tf.nn.l2_loss(y - y_) )

    train_step = tf.train.AdamOptimizer(learning_rate=0.5).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    batch_size = 200
    batch_generator = BatchGenerator(data_x, batch_size=batch_size)
    for _ in range(10000):
        batch = batch_generator.get_batch()
        train_step.run(feed_dict={x: batch, y_: batch})

    accuracy = loss.eval(feed_dict={x: test_x_noisy, y_: test_x})

    print("Accuracy: ", accuracy)


width = test_x_noisy.shape[1]
factor = 0.5

denoiser(width, int(width * factor), noise_level)
