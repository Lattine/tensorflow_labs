# -*- coding: utf-8 -*-

# @Time    : 2019/8/20
# @Author  : Lattine

# ======================
import tensorflow as tf
from .base import BaseModel


class RCNN(BaseModel):
    def __init__(self, config, vocab_size, word_vecters=None):
        super(RCNN, self).__init__(config)
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vecters

        self.build_graph()
        self.init_saver()

    def build_graph(self):
        # 词嵌入层
        with tf.name_scope("embedding"):
            if self.word_vectors is not None:
                embeddings_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"), name="embeddings_w")
            else:
                embeddings_w = tf.get_variable("embedding/embeddings_w", shape=[self.vocab_size, self.config.embedding_dim], initializer=tf.contrib.layers.xavier_initializer())
            embedded_inputs = tf.nn.embedding_lookup(embeddings_w, self.inputs)
            embedded_inputs_bk = embedded_inputs  # 备份一份输入

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("BiLSTM"):
            for idx, hidden_size in enumerate(self.config.hidden_sizes):
                with tf.name_scope("BiLSTM-{}".format(idx)):
                    # 定义前/反向LSTM结构
                    fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True), output_keep_prob=self.keep_prob)
                    bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True), output_keep_prob=self.keep_prob)

                    # 采用动态rnn，可以动态的输入序列长度，若没有输入，则取序列的全长
                    # outputs是一个元组(output_fw, output_bw),其中两个元素的维度都是[batch_size, max_time, hidden_size],
                    # fw 和 bw 的hidden_size一样
                    # current_state 是最终的状态，是个二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, current_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_inputs, dtype=tf.float32, scope="bilstm-{}".format(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]， 并将拼接结果作为下一层输入
                    embedded_inputs = tf.concat(outputs_, axis=2)

            # 将最终输出结果，拆分为前向和后向两个
            output_fw, output_bw = tf.split(embedded_inputs, 2, axis=-1)

            with tf.name_scope("context"):
                shape = [tf.shape(output_fw)[0], 1, tf.shape(output_fw)[2]]
                context_left = tf.concat([tf.zeros(shape), output_fw[:, :-1]], axis=1, name="context_left")
                context_right = tf.concat([output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

            # 将前向，后向的输出和最早的词向量拼接在一起得到最终的词表征
            with tf.name_scope("word_representation"):
                word_repr = tf.concat([context_left, embedded_inputs_bk, context_right], axis=2)
                word_size = self.config.hidden_sizes[-1] * 2 + self.config.embedding_dim

            with tf.name_scope("text_representation"):
                output_size = self.config.model_output_size
                text_w = tf.Variable(tf.random_uniform([word_size, output_size], -1.0, 1.0), name="w2")
                text_b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="b2")

                # tf.einsum可以指定维度的消除运算
                text_repr = tf.tanh(tf.einsum('aij,jk->aik', word_repr, text_w) + text_b)

            # 做max-pool的操作，将时间步的维度消失
            output = tf.reduce_max(text_repr, axis=1)

            # 全连接层的输出
            with tf.name_scope("output"):
                output_w = tf.get_variable("output/output_w", shape=[output_size, self.config.num_classes], initializer=tf.contrib.layers.xavier_initializer())
                output_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="output_b")
                self.l2_loss += tf.nn.l2_loss(output_w)
                self.logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
                self.predictions = self.get_predictions()

            self.loss = self.calculate_loss()
            self.train_op, self.summary_op = self.get_train_op()

            # Metrics
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
            self.metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def get_metrics(self, sess, batch):
        feed_dict = {
            self.inputs: batch['x'],
            self.labels: batch['y'],
            self.keep_prob: 1.0
        }
        return sess.run(self.metric, feed_dict=feed_dict)
