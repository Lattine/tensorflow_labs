# -*- coding: utf-8 -*-

# @Time    : 2019/7/31
# @Author  : Lattine

# ======================

import tensorflow as tf
from .base import BaseModel


class TextCNN(BaseModel):
    def __init__(self, config, vocab_size, word_vectors=None):
        super(TextCNN, self).__init__(config)
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors

        self.build_model()  # 构建计算图
        self.init_saver()  # 初始化Saver对象

    def build_model(self):
        # ----- 词向量层 -----
        with tf.name_scope("embedding"):
            # 加载已有的词向量，或者随机初始化词向量
            if self.word_vectors:
                embeddings_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name='word2vec'), name='embeddings_w')
            else:
                embeddings_w = tf.get_variable('embeddings_w', shape=[self.vocab_size, self.config.embedding_size], initializer=tf.contrib.layers.xavier_initializer())
            # 使用词向量将输入的数据索引转换为词向量表示, [batch_size, sequence_length, embedding_size]
            embedded_words = tf.nn.embedding_lookup(embeddings_w, self.inputs)
            # 卷积的输入是[batch_size, width, height, channel]，因此需要增加维度
            embedded_words_expand = tf.expand_dims(embedded_words, -1)  # [batch_size, sequence_length, embedding_size, 1] = [b, h, w, in]

        # ----- 卷积层，池化层 -----
        pooled_outputs = []
        # 原始论文中使用三种filter(3,4,5),故TextCNN是个多通道单层卷积的模型，可以看作三个单层均价模型的融合
        # for filter_size in self.config.filter_sizes:
        #     with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
        #         # 卷积层，卷积核的大小为 filter_size * embedding_size
        #         filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]  # [height, width, in_channel, out_channel]
        #         conv_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='conv_w')
        #         conv_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name='conv_b')
        #         conv = tf.nn.conv2d(embedded_words_expand, conv_w, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        #         h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name='relu')  # 非线性激活
        #         # 最大池化，每个filter取最大值
        #         pooled = tf.nn.max_pool(h, ksize=[1, self.config.sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
        #         pooled_outputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
                conv_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_w")
                conv_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="conv_b")
                conv = tf.nn.conv2d(
                    embedded_words_expand,
                    conv_w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # relu函数的非线性映射
                h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name="relu")
                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.sequence_length - filter_size + 1, 1, 1],
                    # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)  # CNN网络的输出维度
        h_pool = tf.concat(pooled_outputs, 3)  # 池化后的维度不变，按照最后的维度channel来concat数据
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # 将拼接的数据拉平

        # ----- Dropout -----
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)

        # ----- Output -----
        with tf.name_scope("output"):
            output_w = tf.get_variable("output_w", shape=[num_filters_total, self.config.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='output_b')
            self.l2_loss += tf.nn.l2_loss(output_w)
            # self.l2_loss += tf.nn.l2_loss(output_b) # 偏移是否需要加入约束

            self.logits = tf.nn.xw_plus_b(h_drop, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        # ----- Loss ------
        self.loss = self.calculate_loss() + self.config.l2_reg_lambda * self.l2_loss

        # ----- Train OP -----
        self.train_op, self.summary_op = self.get_train_op()
