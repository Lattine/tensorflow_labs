# -*- coding: UTF-8 -*-
# @Auther: Lattine
# @Date: 2019-05-28

import tensorflow as tf 
from parameter import Parameters as pm

class TextCNN(object):
    name = 'textcnn'
    def __init__(self, pm=pm):
        self.pm = pm

        self.input_x = tf.placeholder(tf.int32, [None, self.pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.pm.num_tags], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self._cnn()
    
    def _cnn(self):
        with tf.device('/cpu:0'), tf.name_scope('Embedding'):
            embedding = tf.get_variable('embedding', [self.pm.vocab_size, self.pm.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        
        with tf.name_scope('CNN'):
            conv = tf.layers.conv1d(embedding_inputs, self.pm.num_filters, self.pm.kernel_size, name='conv')
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        
        with tf.name_scope('Score'):
            fc = tf.layers.dense(gmp, self.pm.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            self.logits = tf.layers.dense(fc, self.pm.num_tags, name='fc2')
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)
        
        with tf.name_scope('Optimizer'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
        
        with tf.name_scope('Accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    def feed_data(self, x, y, keep_prob=1.0):
        feed = {
            self.input_x:x,
            self.input_y:y,
            self.keep_prob: keep_prob
        }
        return feed
        