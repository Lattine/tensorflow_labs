# -*- coding: utf-8 -*-

# @Time    : 2019/7/30
# @Author  : Lattine

# ======================


class DataBase:
    def __init__(self, config):
        self.cfg = config

    def read_data(self):
        """读取数据"""
        raise NotImplementedError

    @staticmethod
    def trans_w2ix(inputs, w2ix):
        """数据转为索引"""
        raise NotImplementedError

    def padding(self, inputs, sequence_length):
        """序列填充"""
        raise NotImplementedError

    def gen_data(self):
        """生成可导入模型的数据"""
        raise NotImplementedError

    def next_batch(self, x, y, batch_size):
        """生成批次数据"""
        raise NotImplementedError


class TrainDataBase(DataBase):
    def __init__(self, config):
        super(TrainDataBase, self).__init__(config)

    def read_data(self):
        """读取数据"""
        raise NotImplementedError

    def remove_stopwords(self, inputs):
        """去除低频词和停用词"""
        raise NotImplementedError

    def get_word_vectors(self, vocab):
        """加载词向量，并获得相应的词向量矩阵"""
        raise NotImplementedError

    def gen_vocab(self, words, labels):
        """生成词汇表"""
        raise NotImplementedError

    @staticmethod
    def trans_w2ix(inputs, w2ix):
        """数据转为索引"""
        raise NotImplementedError

    def padding(self, inputs, sequence_length):
        """序列填充"""
        raise NotImplementedError

    def gen_data(self):
        """生成可导入模型的数据"""
        raise NotImplementedError

    def next_batch(self, x, y, batch_size):
        """生成批次数据"""
        raise NotImplementedError


class TestDataBase(DataBase):
    def __init__(self, config):
        super(TestDataBase, self).__init__(config)

    def read_data(self):
        """读取数据"""
        raise NotImplementedError

    def load_vocab(self):
        """加载词汇表"""
        raise NotImplementedError

    @staticmethod
    def trans_w2ix(inputs, w2ix):
        """数据转为索引"""
        raise NotImplementedError

    def padding(self, inputs, sequence_length):
        """序列填充"""
        raise NotImplementedError

    def gen_data(self):
        """生成可导入模型的数据"""
        raise NotImplementedError

    def next_batch(self, x, y, batch_size):
        """生成批次数据"""
        raise NotImplementedError
