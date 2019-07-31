# -*- coding: utf-8 -*-

# @Time    : 2019/7/30
# @Author  : Lattine

# ======================
import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config

        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None], name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.l2_loss = tf.constant(0.0)
        self.loss = 0.0
        self.train_op = None
        self.summary_op = None
        self.logits = None
        self.predictions = None
        self.saver = None

    def calculate_loss(self):
        """计算损失，支持二分类，多分类"""
        with tf.name_scope("loss"):
            losses = 0.0
            if self.config.num_classes == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.labels, [-1, 1]))
            elif self.config.num_classes > 1:
                self.labels = tf.cast(self.labels, dtype=tf.int32)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            loss = tf.reduce_mean(losses)
            return loss

    def get_optimizer(self):
        """获取优化器"""
        if self.config.optimization == "adam":
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        elif self.config.optimization == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
        return optimizer

    def get_train_op(self):
        """获取训练入口"""
        optimizer = self.get_optimizer()
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip_grad)
        train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

        tf.summary.scalar("loss", self.loss)
        summary_op = tf.summary.merge_all()

        return train_op, summary_op

    def get_predictions(self):
        predictions = None
        if self.config.num_classes == 1:
            predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name='predictions')
        elif self.config.num_classes > 1:
            predictions = tf.argmax(self.logits, axis=-1, name='predictions')
        return predictions

    def build_model(self):
        """构建图结构"""
        raise NotImplementedError

    def init_saver(self):
        """初始化Saver对象"""
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, dropout_prob):
        """训练模型"""
        feed_dict = {
            self.inputs: batch['x'],
            self.labels: batch['y'],
            self.keep_prob: dropout_prob
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

    def infer(self, sess, inputs):
        """预测新数据"""
        feed_dict = {
            self.inputs: inputs,
            self.keep_prob: 1.0
        }
        prediction = sess.run(self.predictions, feed_dict=feed_dict)
        return prediction
