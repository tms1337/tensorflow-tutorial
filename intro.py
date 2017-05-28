import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


def main():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)

    print(node1, node2)

    sess = tf.Session()

    print(sess.run([node1, node2]))
    print(sess.run(node1))

    node3 = tf.add(node1, node2)
    print(sess.run(node3))

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)

    print(a, b)

    c = a + b

    print(sess.run(c, {a: 3, b: 4.5}))

    W = tf.Variable([-3.0], tf.float32)
    b = tf.Variable([-0.4], tf.float32)

    x = tf.placeholder(tf.float32)

    linear_model = W * x + b

    init = tf.global_variables_initializer()
    sess.run(init)

    result = sess.run(linear_model, {x: 2.0})
    print(result)

    result = sess.run(linear_model, {x: [1, 2, 3, 4]})

    print(result)

    y = tf.placeholder(tf.float32)

    deltas_sq = tf.square(y - linear_model)
    loss = tf.reduce_sum(deltas_sq)

    fix_W = tf.assign(W, [1.0])
    fix_b = tf.assign(b, [-1.0])
    sess.run([fix_W, fix_b])

    io = {x: [1, 2, 3, 4], y: [1, 2, 3, 4]}
    result = sess.run(loss, io)
    print(result)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    print(sess.run([W, b]))

    for i in range(10):
        sess.run(train, io)


    print(sess.run([W, b]))

def contrib():
    features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4, num_epochs=1000)

    estimator.fit(input_fn=input_fn, steps=1000)
    print(estimator.evaluate(input_fn=input_fn))

if __name__ == "__main__":
    contrib()
