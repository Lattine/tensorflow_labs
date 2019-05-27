# -*- coding: UTF-8 -*-
# @Auther: Lattine
# @Date: 2019-05-25

import tensorflow as tf 
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from parameter import Parameters as pm

class BiLSTM_CRF:
    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, None], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self._bilstm_crf()
    
    def _bilstm_crf(self):
        with tf.device('/cpu:0'), tf.name_scope('Embedding'):
            embedding = tf.Variable(tf.truncated_normal([pm.vocab_size, pm.embedding_size], -0.25, 0.25), name='embedding')
            embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)
            self.embedding = tf.nn.dropout(embedding_input, keep_prob=self.keep_prob)
        
        with tf.name_scope('Cell'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim, state_is_tuple=True)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, self.keep_prob)

            cell_bw = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, self.keep_prob)
        
        with tf.name_scope('BiLSTM'):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.embedding, sequence_length=self.seq_length, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)
        
        with tf.name_scope('Output'):
            s = tf.shape(outputs)
            output = tf.reshape(outputs, [-1, 2*pm.hidden_dim])
            output = tf.layers.dense(output, pm.num_tags)
            output = tf.contrib.layers.dropout(output, self.keep_prob)
            self.logits = tf.reshape(output, [-1, s[1], pm.num_tags])
        
        with tf.name_scope('Crf'):
            self.log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.input_y, sequence_lengths=self.seq_length)
            # log_likelihood是对数似然函数，transition_params是转移概率矩阵
            # crf_log_likelihood{inputs:[batch_size,max_seq_length,num_tags],
            # tag_indices:[batch_size,max_seq_length],
            # sequence_lengths:[real_seq_length]
            # transition_params: A [num_tags, num_tags] transition matrix
            # log_likelihood: A scalar containing the log-likelihood of the given sequence of tag indices.

        with tf.name_scope('Loss'):
            self.loss = tf.reduce_mean(-self.log_likelihood)
        
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step) # global_step 自动+1

    def feed_data(self, x_batch, y_batch, seq_length, keep_prob):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.seq_length: seq_length,
            self.keep_prob:keep_prob
        }
        return feed_dict
    
    def test(self, sess, processor, x, y):
        batch = processor.batch_iter(x, y, pm.batch_size)
        for x_batch, y_batch in batch:
            x_batch, x_length = processor.process(x_batch)
            y_batch, y_length = processor.process(y_batch)
            feed_dict = self.feed_data(x_batch, y_batch, x_length, 1.0)
            loss = sess.run(self.loss, feed_dict=feed_dict)
        return loss
    
    def predict(self, sess, processor, x_batch):
        seq_pad, seq_length = processor.process(x_batch)
        logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict={self.input_x:seq_pad, self.seq_length:seq_length, self.keep_prob:1.0})
        label = []
        for logit, length in zip(logits, seq_length):
            # logit 是每个句子的输出值，length是句子的真实长度， 调用viterbi求解最优标注序列
            viterbi_seq, _ = viterbi_decode(logit[:length], transition_params)
            label.append(viterbi_seq)
        return label