import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")


def conv_layer(conv1_feature_n, patch_size, x_image, input_channels):
    W = weight_var([patch_size[0], patch_size[1], input_channels, conv1_feature_n])
    b = bias_var([conv1_feature_n])
    h = tf.nn.relu(conv2d(x_image, W) + b)
    pool = pool2x2(h)

    return pool


if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h1 = conv_layer(32, [5, 5], x_image, 1)
    h2 = conv_layer(64, [5, 5], h1, 32)

    h2_flat = tf.reshape(h2, [-1, 7*7*64])

    w_fc = weight_var([7 * 7 * 64, 1024])
    b_fc = bias_var([1024])
    full = tf.nn.relu(tf.matmul(h2_flat, w_fc) + b_fc)



