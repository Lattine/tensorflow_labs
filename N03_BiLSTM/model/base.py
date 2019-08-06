# -*- coding: utf-8 -*-

# @Time    : 2019/8/5
# @Author  : Lattine

# ======================
import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config

        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inputs')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, config.num_classes], name='labels')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        self.l2_loss = tf.constant(0.0)
        self.loss = 0.0
        self.train_op = None
        self.summary_op = None
        self.logits = None
        self.predictions = None
        self.saver = None

    def build_graph(self):
        raise NotImplementedError

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def calculate_loss(self):
        """计算损失"""
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels))
            return loss

    def get_optimizer(self):
        """指定特定的优化器"""
        if self.config.optimization == "adam":
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        elif self.config.optimization == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
        return optimizer

    def get_train_op(self):
        optimizer = self.get_optimizer()
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip_grad)
        train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

        tf.summary.scalar('loss', self.loss)
        summary_op = tf.summary.merge_all()

        return train_op, summary_op

    def get_predictions(self):
        predictions = tf.argmax(self.logits, axis=-1, name='predictions')
        return predictions

    def train(self, sess, batch, drop):
        feed_dict = {
            self.inputs: batch['x'],
            self.labels: batch['y'],
            self.keep_prob: drop
        }
        _, summary, loss, predictions = sess.run([self.train_op, self.summary_op, self.loss, self.predictions], feed_dict=feed_dict)
        return summary, loss, predictions

    def eval(self, sess, batch):
        """验证模型"""
        feed_dict = {
            self.inputs: batch['x'],
            self.labels: batch['y'],
            self.keep_prob: 1.0
        }
        summary, loss, predictions = sess.run([self.summary_op, self.loss, self.predictions], feed_dict=feed_dict)
        return summary, loss, predictions

    def predict(self, sess, inputs):
        """预测新数据"""
        feed_dict = {
            self.inputs: inputs,
            self.keep_prob: 1.0
        }
        prediction = sess.run(self.predictions, feed_dict=feed_dict)
        return prediction

    def get_metrics(self, sess, batch):
        raise NotImplementedError
