import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    batch_n = 1000
    for i in range(batch_n):
        if i % 100 == 0:
            print("Training batch %d / %d" % (i, batch_n))
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("Accuracy: %f" % (100.0 * result))