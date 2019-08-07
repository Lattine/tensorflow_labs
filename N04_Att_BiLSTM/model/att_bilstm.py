# -*- coding: utf-8 -*-

# @Time    : 2019/8/7
# @Author  : Lattine

# ======================
import tensorflow as tf
from .base import BaseModel


class AttBiLSTM(BaseModel):
    def __init__(self, config, vocab_size, word_vectors=None):
        super(AttBiLSTM, self).__init__(config)
        self.config = config

        self.vocab_size = vocab_size
        self.word_vectors = word_vectors

        self.build_graph()
        self.init_saver()

    def build_graph(self):
        with tf.name_scope("embedding"):
            if self.word_vectors is not None:
                embeddings_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"), name="embeddings_w")  # 以预训练的词向量初始化
            else:
                embeddings_w = tf.get_variable("embeddings_w", shape=[self.vocab_size, self.config.embedding_dim], initializer=tf.contrib.layers.xavier_initializer())
            lstm_inputs = tf.nn.embedding_lookup(embeddings_w, self.inputs)

        # BiLSTM层，深度由len(hidden_sizes)控制
        with tf.name_scope("BiLIST"):
            for idx, hidden_size in enumerate(self.config.hidden_sizes):
                with tf.name_scope("BiLSTM-{}".format(idx)):
                    fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True), output_keep_prob=self.keep_prob)
                    bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True), output_keep_prob=self.keep_prob)

                    # 采用动态RNN，可以动态的输入序列长度，如果没有输入，则取序列的全长
                    # outputs = (outputs_fw, outputs_bw), 维度都是[batch_size, max_time, hidden_size]
                    # current_state 是最终状态，二元组(state_fw, state_bw); 其中state_f*为[batch_size, s], s=(h,c)
                    outputs_, current_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, lstm_inputs, dtype=tf.float32, scope="bilstm-{}".format(idx))

                    # 合并outputs_fw、outputs_bw，作为下一层BiLSTM的输入
                    lstm_inputs = tf.concat(outputs_, 2)  # [batch_size, max_time, hidden_size*2]

        # 将BiLSTM层的最后一层输出，切分为前向、后向
        outputs = tf.split(lstm_inputs, 2, -1)

        # 论文将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]
            attention_output = self._attention(H)
            output_size = self.config.hidden_sizes[-1]

        # 全连接层的输出
        with tf.name_scope("Output"):
            output_w = tf.get_variable("output_w", shape=[output_size, self.config.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.logits = tf.nn.xw_plus_b(attention_output, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        self.loss = self.calculate_loss()
        self.loss += self.config.l2_reg_lambda * self.l2_loss
        self.train_op, self.summary_op = self.get_train_op()

        # Metrics
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def get_metrics(self, sess, batch):
        feed_dict = {
            self.inputs: batch["x"],
            self.labels: batch["y"],
            self.keep_prob: 1.0
        }
        acc = sess.run(self.accuracy, feed_dict=feed_dict)
        return acc

    def _attention(self, H):
        """ 利用Attention机制得到句子的词向量表示"""
        hidden_size = self.config.hidden_sizes[-1]  # 获取最后一层LSTM神经元个数
        W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))  # 初始化一个权重向量， 是可以被训练的参数
        M = tf.tanh(H)  # 对BiLSTM的输出，用激活函数做非线性转换
        # 对W和M做矩阵运算， M=[batch_size, time_step, hidden_size]，计算前：维度转换为[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1], 每一个时间步的输出用向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

        restoreM = tf.reshape(newM, [-1, self.config.sequence_length])  # newM转换为[batch_size, time_step]

        self.alpha = tf.nn.softmax(restoreM)  # 用softmax做归一化处理[batch_size, time_step]

        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.sequence_length, 1]))  # 利用alpha的值对H进行加权求和，用矩阵运算直接操作
        sequeezeR = tf.squeeze(r)  # 将三维压缩成二维[batch_size, hidden_size]
        sentenceRepren = tf.tanh(sequeezeR)

        output = tf.nn.dropout(sentenceRepren, self.keep_prob)  # 对Attention的输出做dropout处理
        return output
