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
batch_size = 64


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
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # FC2
    W_fc2 = weight_variable([1024, 10])
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

# ------------------------ Result --------------------------
# Epoch 0 :Train loss: 0.09977313876152039, Test loss: 0.09497814625501633, Train acc: 0.9704727530479431, Test acc: 0.9695000052452087
# Epoch 1 :Train loss: 0.05402466282248497, Test loss: 0.054240208119153976, Train acc: 0.9837818145751953, Test acc: 0.9822999835014343
# Epoch 2 :Train loss: 0.03888505697250366, Test loss: 0.042723070830106735, Train acc: 0.9878363609313965, Test acc: 0.9854000210762024
# Epoch 3 :Train loss: 0.027384402230381966, Test loss: 0.033973149955272675, Train acc: 0.9917272925376892, Test acc: 0.9879999756813049
# Epoch 4 :Train loss: 0.022167138755321503, Test loss: 0.029579292982816696, Train acc: 0.9934545159339905, Test acc: 0.9883999824523926
# Epoch 5 :Train loss: 0.017373420298099518, Test loss: 0.031052500009536743, Train acc: 0.9947636127471924, Test acc: 0.9887999892234802
# Epoch 6 :Train loss: 0.018054625019431114, Test loss: 0.03300003707408905, Train acc: 0.9942181706428528, Test acc: 0.9890999794006348
# Epoch 7 :Train loss: 0.011205444112420082, Test loss: 0.02524348720908165, Train acc: 0.9968545436859131, Test acc: 0.9914000034332275
# Epoch 8 :Train loss: 0.0111664654687047, Test loss: 0.027720559388399124, Train acc: 0.9962000250816345, Test acc: 0.9901999831199646
# Epoch 9 :Train loss: 0.00872449018061161, Test loss: 0.025138182565569878, Train acc: 0.9976727366447449, Test acc: 0.9919000267982483