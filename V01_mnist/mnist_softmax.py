#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ====== INFO ======
# @File  : mnist_softmax.py
# @Author: Lattine
# @Date  : 2019/7/13 上午7:22
# @Desc  : 

# ==================

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset
mnist = input_data.read_data_sets('./data_mnist/', one_hot=True)

# Setting
features = 784
classes = 10

learning_rate = 0.01
epoches = 10
batch_size = 100

# Build Graph
g = tf.Graph()
with g.as_default() as g:
    x = tf.placeholder(tf.float32, [None, features], name='features')
    y_ = tf.placeholder(tf.float32, [None, classes], name='targets')

    W = tf.Variable(tf.zeros([features, classes]))
    b = tf.Variable(tf.zeros([classes]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training & Validation
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoches):
        total_batches = mnist.train.num_examples // batch_size
        for ii in range(total_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        loss_train = sess.run(cross_entropy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
        loss_test = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        acc_train = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
        acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print(f"Epoch {i} :Train loss: {loss_train}, Test loss: {loss_test}, Train acc: {acc_train}, Test acc: {acc_test}")

# ------------------------ Result --------------------------
# Epoch 0 :Train loss: 18187.76171875, Test loss: 3303.51513671875, Train acc: 0.9025454521179199, Test acc: 0.9031000137329102
# Epoch 1 :Train loss: 16865.87890625, Test loss: 3117.572265625, Train acc: 0.910945475101471, Test acc: 0.9092000126838684
# Epoch 2 :Train loss: 15485.51171875, Test loss: 2869.690673828125, Train acc: 0.9206181764602661, Test acc: 0.9194999933242798
# Epoch 3 :Train loss: 15061.9833984375, Test loss: 2853.0009765625, Train acc: 0.9226545691490173, Test acc: 0.9182999730110168
# Epoch 4 :Train loss: 15600.0712890625, Test loss: 3039.84765625, Train acc: 0.918109118938446, Test acc: 0.9110999703407288
# Epoch 5 :Train loss: 15619.396484375, Test loss: 3079.95458984375, Train acc: 0.919981837272644, Test acc: 0.9142000079154968
# Epoch 6 :Train loss: 15076.638671875, Test loss: 2916.74609375, Train acc: 0.9213272929191589, Test acc: 0.9172999858856201
# Epoch 7 :Train loss: 14678.474609375, Test loss: 2849.71728515625, Train acc: 0.923872709274292, Test acc: 0.9200999736785889
# Epoch 8 :Train loss: 14142.984375, Test loss: 2774.58447265625, Train acc: 0.9282363653182983, Test acc: 0.9230999946594238
# Epoch 9 :Train loss: 14074.091796875, Test loss: 2835.330810546875, Train acc: 0.9289090633392334, Test acc: 0.9229999780654907
