# -*- coding: utf-8 -*-

# @Time    : 2019/8/5
# @Author  : Lattine

# ======================

import tensorflow as tf

from .base import BaseModel


class BiLSTM(BaseModel):
    def __init__(self, config, vocab_size, embeddings_vectors=None):
        super(BiLSTM, self).__init__(config)
        self.config = config
        self.vocab_size = vocab_size
        self.embeddings_vectors = embeddings_vectors

        self.build_graph()
        self.init_saver()

    def build_graph(self):
        # 词嵌入层
        with tf.name_scope("embedding"):
            if self.embeddings_vectors is not None:
                embeddings_w2v = tf.Variable(tf.cast(self.embeddings_vectors, dtype=tf.float32, name='word2vec'), name='embeddings_w')
            else:
                embeddings_w2v = tf.get_variable("embedding/embeddings_w", shape=[self.vocab_size, self.config.embedding_dim], initializer=tf.contrib.layers.xavier_initializer())
            embedded_inputs = tf.nn.embedding_lookup(embeddings_w2v, self.inputs)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("BiLSTM"):
            for idx, hidden_size in enumerate(self.config.hidden_sizes):
                with tf.name_scope("BiLSTM-{}".format(idx)):
                    # 定义前/反向LSTM结构
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True), output_keep_prob=self.keep_prob)
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True), output_keep_prob=self.keep_prob)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
                    # fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_inputs, dtype=tf.float32, scope='bilstm-{}'.format(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                    concat_outputs = tf.concat(outputs, 2)

        # 取出最后的时间步输出，作为全连接的输入
        final_output = concat_outputs[:, -1, :]

        final_output_size = self.config.hidden_sizes[-1] * 2  # 因为是双向LSTM，最终的输出是fw和bw的拼接，所以维度要乘以2
        final_output_flat = tf.reshape(final_output, [-1, final_output_size])  # Reshape

        # 全连接层输出
        with tf.name_scope('output'):
            output_w = tf.get_variable('output_w', shape=[final_output_size, self.config.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.logits = tf.nn.xw_plus_b(final_output_flat, output_w, output_b)
            self.predictions = self.get_predictions()

        self.loss = self.calculate_loss()
        self.loss += self.config.l2_reg_lambda * self.l2_loss
        self.train_op, self.summary_op = self.get_train_op()

        # Metrics
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def get_metrics(self, sess, batch):
        feed_dict = {
            self.inputs: batch['x'],
            self.labels: batch['y'],
            self.keep_prob: 1.0
        }
        return sess.run(self.accuracy, feed_dict=feed_dict)
