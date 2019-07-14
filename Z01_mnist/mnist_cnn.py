#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ====== INFO ======
# @File  : mnist_cnn.py
# @Author: Lattine
# @Date  : 2019/7/13 上午8:00
# @Desc  : 

# ==================

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset
mnist = input_data.read_data_sets('./data_mnist/', one_hot=True)

# Setting
features = 784
classes = 10
feature_row = 28
feature_col = 28
feature_channel = 1
keep_prob_train = 0.5

learning_rate = 1e-4
epoches = 10
batch_size = 16


# ------------------ Function Begin --------------------
def weight_variable(shape):
    init = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2X2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# ------------------ Function End --------------------

# Build Graph
g = tf.Graph()
with g.as_default() as g:
    # Inputs
    x = tf.placeholder(tf.float32, [None, features])
    x_image = tf.reshape(x, [-1, feature_row, feature_col, feature_channel])
    y_ = tf.placeholder(tf.float32, [None, classes])
    keep_prob = tf.placeholder(tf.float32)

    # Conv1 with activation
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2D(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2X2(h_conv1)

    # Conv2 with activation
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2D(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2X2(h_conv2)

    # FC1 with activation, dropout
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    W_fc1 = weight_variable([7 * 7 * 64, 512])
    b_fc1 = bias_variable([512])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # FC2
    W_fc2 = weight_variable([512, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Loss (cross entropy)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train & Validation
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoches):
        total_batches = mnist.train.num_examples // batch_size
        for ii in range(total_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _ = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: keep_prob_train})
        loss_train = sess.run(cross_entropy, feed_dict={x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0})
        loss_test = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        acc_train = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0})
        acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print(f"Epoch {i} :Train loss: {loss_train}, Test loss: {loss_test}, Train acc: {acc_train}, Test acc: {acc_test}")
