# -*- coding: utf-8 -*-

# @Time    : 2019/7/30
# @Author  : Lattine

# ======================
import os
import pickle
from collections import Counter

import numpy as np
import gensim

from .base import TrainDataBase


class TrainData(TrainDataBase):
    def __init__(self, config):
        super(TrainData, self).__init__(config)

        self._train_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config.train_data)
        self._output_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config.output_path)
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        self._word_vectors_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config.word_vectors_path) if config.word_vectors_path else None
        self._stopwords_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config.stopwords) if config.stopwords else None

        self._sequence_length = config.sequence_length
        self._batch_size = config.batch_size
        self._embedding_size = config.embedding_size
        self._vocab_size = config.vocab_size
        self.word_vectors = None

    def read_data(self):
        """
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
        """
        inputs, labels = [], []
        with open(self._train_data_path, 'r', encoding='utf8') as fr:
            for line in fr:
                try:
                    text, label = line.strip().split("<SEP>")
                    inputs.append(text.strip().split(" "))
                    labels.append(label)
                except:
                    continue
        return inputs, labels

    def remove_stopwords(self, inputs):
        """ 去除低频词和停用词"""
        # 统计词频
        word_counts = Counter()
        for sent in inputs:
            word_counts.update(sent)

        # 去除低频词
        words = []
        for k, v in word_counts.most_common(self._vocab_size - 4):  # 统计最常用的词，为词表大小减去<START>,<UNK>,<PAD>,<END>
            words.append(k)

        # 如果设置停用词表，去除停用词
        if self._stopwords_path:
            with open(self._stopwords_path, 'r', encoding="utf8") as fr:
                stopwords = [line.strip() for line in fr]
            words = [w for w in words if w not in stopwords]

        return words

    def get_word_vectors(self, vocab):
        """加载词向量，并获得相应的词向量矩阵"""
        word_vectors = (1 / np.sqrt(len(vocab)) * (2 * np.random.rand(len(vocab), self._embedding_size) - 1))  # 有待深究
        if os.path.splitext(self._word_vectors_path)[-1] == ".bin":
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._word_vectors_path, binary=True)
        else:
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._word_vectors_path)
        for i in range(len(vocab)):
            try:
                vector = word_vec.wv[vocab[i]]
                word_vectors[i, :] = vector
            except:
                print(f"{vocab[i]} not in w2v file.")
        return word_vectors

    def gen_vocab(self, words, labels):
        """
        生成词汇表"
        :param words:  训练集所过滤的词汇列表
        :param labels: 标签
        :return: word2vec, label2vec
        """
        word_vectors_path = os.path.join(self._output_path, 'word_vectors.npy')
        word_to_index_path = os.path.join(self._output_path, 'word_to_index.pkl')
        label_to_index_path = os.path.join(self._output_path, 'label_to_index.pkl')

        # 如果已有词向量，则直接加载
        if os.path.exists(word_vectors_path):
            print("load word_vectors.")
            self.word_vectors = np.load(word_vectors_path)
        # 如果存在词汇表，则直接加载
        if os.path.exists(word_to_index_path) and os.path.exists(label_to_index_path):
            print("load word_to_index")
            with open(word_to_index_path, 'rb') as fr:
                word_to_index = pickle.load(fr)
            print("load label to index")
            with open(label_to_index_path, 'rb') as fr:
                label_to_index = pickle.load(fr)
            self.vocab_size = len(word_to_index)

            return word_to_index, label_to_index

        words = ["<PAD>", "<UNK>"] + words
        vocab = words[:self.vocab_size]  # 词汇表上限

        # 如果vocab的长度小于config设置的值，则用实际长度
        self.vocab_size = len(vocab)
        if self._word_vectors_path:
            word_vectors = self.get_word_vectors(vocab)
            self.word_vectors = word_vectors
            np.save(word_vectors_path, self.word_vectors)  # 将数据集相关的词向量存入特定目录

        word_to_index = {w: i for i, w in enumerate(vocab)}

        # 将词汇-索引字典保存
        with open(word_to_index_path, 'wb') as fw:
            pickle.dump(word_to_index, fw)

        # 将标签-索引字典保存
        unique_labels = list(set(labels))
        label_to_index = {w: i for i, w in enumerate(unique_labels)}
        with open(label_to_index_path, 'wb') as fw:
            pickle.dump(label_to_index, fw)

        return word_to_index, label_to_index

    @staticmethod
    def trans_w2ix(inputs, w2ix):
        """数据转为索引"""
        inputs_idx = [[w2ix.get(w, w2ix.get("<UNK>")) for w in sentence] for sentence in inputs]
        return inputs_idx

    @staticmethod
    def trans_t2ix(labels, t2ix):
        labels_idx = [[t2ix.get(label)] for label in labels]
        return labels_idx

    def padding(self, inputs, sequence_length):
        """序列填充/截断"""
        new_inputs = [sentence[:sequence_length] if len(sentence) > sequence_length else sentence + [0] * (sequence_length - len(sentence)) for sentence in inputs]
        return new_inputs

    def gen_data(self):
        """生成可导入模型的数据"""
        train_data_path = os.path.join(self._output_path, "train_data.pkl")
        label_to_index_path = os.path.join(self._output_path, "label_to_index.pkl")
        word_to_index_path = os.path.join(self._output_path, "word_to_index.pkl")
        word_vectors_path = os.path.join(self._output_path, "word_vectors.npy")

        # 如果存在，则直接加载
        if os.path.exists(train_data_path) and os.path.exists(label_to_index_path) and os.path.exists(word_to_index_path):
            print("load existed train data")
            with open(train_data_path, 'rb') as fr:
                train_data = pickle.load(fr)
            with open(word_to_index_path, 'rb') as fr:
                word_to_index = pickle.load(fr)
            with open(label_to_index_path, 'rb') as fr:
                label_to_index = pickle.load(fr)
            self.vocab_size = len(word_to_index)

            # 尝试加载词向量
            if os.path.exists(word_vectors_path):
                self.word_vectors = np.load(word_vectors_path)
            return np.array(train_data['inputs_idx']), np.array(train_data['labels_idx']), label_to_index

        # --------- 原始处理流程 ----------
        # 1.读取原始数据
        inputs, labels = self.read_data()
        print("read finished")

        # 2.去除低频词和停用词
        words = self.remove_stopwords(inputs)
        print("word filter process finished")

        # 3.获取词汇-索引字典
        word_to_index, label_to_index = self.gen_vocab(words, labels)
        print("vocab process finished")

        # 4.文本转索引
        inputs_idx = self.trans_w2ix(inputs, word_to_index)
        print("word to index finished")

        # 5.对文本作PADDING
        inputs_idx = self.padding(inputs_idx, self._sequence_length)
        print("padding finished")

        # 6.标签转索引
        labels_idx = self.trans_t2ix(labels, label_to_index)
        print("label to index finished")

        # 构建训练数据字典并入库，以备后续直接加载
        train_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
        with open(train_data_path, 'wb') as fw:
            pickle.dump(train_data, fw)
        return np.array(inputs_idx), np.array(labels_idx), label_to_index

    def next_batch(self, x, y, batch_size):
        """生成批次数据"""
        # 随机化
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

        num_batches = len(x) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_x = np.array(x[start:end], dtype='int64')
            batch_y = np.array(y[start:end], dtype="float32")

            yield batch_x, batch_y
