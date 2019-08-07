# -*- coding: utf-8 -*-

# @Time    : 2019/8/5
# @Author  : Lattine

# ======================

import os
import pickle
from collections import Counter

import gensim
import numpy as np


# 构造数据加载器逻辑
# 1. 读取原始数据，分开文本、标签
# 2. 过滤停用词、低频词
# 3. 训练时，根据过滤后的训练语料，构造出索引字典；测试时，加载索引字典
# 4. 将文本、标签转为索引表示；如果是分类任务，标签需要one-hot操作
# 5. 将文本截断或者PADDING成固定长度
# 6. 文本、标签持久化，返回
class DataBase:
    def __init__(self, config):
        self.config = config
        self._output_path = os.path.join(config.BASE_DIR, config.output_path)
        self._check_directory(self._output_path)  # 保证output目录存在
        self._stopwords_path = os.path.join(config.BASE_DIR, config.stopwords_path) if config.stopwords_path else None
        self._word_vectors_path = os.path.join(config.BASE_DIR, config.word_vectors_path) if config.word_vectors_path else None

        self.vocab_size = config.vocab_size
        self._sequence_length = config.sequence_length
        self._embedding_size = config.embedding_dim
        self.word_vectors = None

    def read_data(self, data_path, split_by="<SEP>"):
        """读取数据"""
        inputs = []
        labels = []
        with open(data_path, 'r', encoding="utf8") as fr:
            for line in fr:
                try:
                    text, label = line.strip().split(split_by)
                    inputs.append(text.strip().split())
                    labels.append(label)
                except:
                    print("Error with line: ", line)
        return inputs, labels

    def remove_stopwords_and_low_frequent(self, inputs):
        """ 去除停用词和低频词，低频词由词表大小控制"""
        # 如果设置停用词表，去除停用词
        if self._stopwords_path:
            with open(self._stopwords_path, 'r', encoding="utf8") as fr:
                stopwords = [line.strip() for line in fr]
                stopwords = set(stopwords)  # 转为SET，加速索引
            inputs = [[w for w in sent if w not in stopwords] for sent in inputs]

        # 统计词频，根据设置的字典大小去除停用词
        words = self._count_words(inputs, self.vocab_size - 4)
        inputs = [[w for w in sent if w in words] for sent in inputs]
        return inputs

    def generate_vocab(self, inputs, labels):
        """生成索引字典"""
        word_to_index_path = os.path.join(self._output_path, 'word_to_index.pkl')
        label_to_index_path = os.path.join(self._output_path, 'label_to_index.pkl')

        words = self._count_words(inputs)
        words = ["<PAD>", "<UNK>"] + list(words)
        vocab = words[:self.vocab_size]  # 词汇表上限
        self.vocab_size = len(vocab)  # 如果vocab的长度小于config设置的值，则用实际长度

        # 将词汇-索引字典保存
        word_to_index = {w: i for i, w in enumerate(vocab)}
        with open(word_to_index_path, 'wb') as fw:
            pickle.dump(word_to_index, fw)

        # 将标签-索引字典保存
        unique_labels = list(set(labels))
        label_to_index = {w: i for i, w in enumerate(unique_labels)}
        with open(label_to_index_path, 'wb') as fw:
            pickle.dump(label_to_index, fw)

        # 加载词向量文件
        if self._word_vectors_path:
            self.word_vectors = self.get_word_vectors(vocab)

        return word_to_index, label_to_index

    def trans_w2ix(self, inputs, w2ix):
        """数据转为索引"""
        inputs_idx = [[w2ix.get(w, w2ix.get("<UNK>")) for w in sentence] for sentence in inputs]
        return inputs_idx

    def trans_t2ix_with_onehot(self, labels, t2ix):
        """标签转索引， 使用np.eye()快速生成One-Hot编码"""
        labels_idx = [t2ix.get(label) for label in labels]
        onehots = np.eye(self.config.num_classes)[labels_idx]
        return onehots.tolist()

    def padding(self, inputs):
        """ 固定长度"""
        inputs_new = [sent[:self._sequence_length] if len(sent) > self._sequence_length else sent + [0] * (self._sequence_length - len(sent)) for sent in inputs]
        return inputs_new

    def next_batch(self, x, y, batch_size):
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

        num_batches = len(x) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            xs = np.array(x[start: end], dtype='int64')
            ys = np.array(y[start: end], dtype='float32')

            yield dict(x=xs, y=ys)

    def get_word_vectors(self, vocab):
        word_vectors = (1 / np.sqrt(len(vocab)) * (2 * np.random.rand(len(vocab), self._embedding_size) - 1))
        if os.path.splitext(self._word_vectors_path)[-1] == ".bin":
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._word_vectors_path, binary=True)
        else:
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._word_vectors_path)

        for i in range(len(vocab)):
            try:
                vector = word_vec.wv[vocab[i]]
                word_vectors[i, :] = vector
            except:
                print(f"{vocab[i]} not in in existed vectors file.")
        return word_vectors

    def _check_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _pickle_dump(self, data, data_path):
        with open(data_path, 'wb') as fw:
            pickle.dump(data, fw)

    def _count_words(self, inputs, size=None):
        word_counts = Counter()
        for sent in inputs:
            word_counts.update(sent)
        words = set()
        for k, v in word_counts.most_common(size):  # 统计最常用的词，为词表大小减去<START>,<UNK>,<PAD>,<END>
            words.add(k)
        return words


class TrainData(DataBase):
    def __init__(self, config):
        super(TrainData, self).__init__(config)

    def gen_train_data(self, data_path):
        """构造可导入模型的数据集"""
        inputs, labels = self.read_data(data_path=data_path)
        inputs = self.remove_stopwords_and_low_frequent(inputs)
        w2ix, t2ix = self.generate_vocab(inputs, labels)
        inputs = self.trans_w2ix(inputs, w2ix)
        labels = self.trans_t2ix_with_onehot(labels, t2ix)
        inputs = self.padding(inputs)

        data_path = os.path.join(self._output_path, 'train_data.pkl')
        data = {'inputs': inputs, 'labels': labels}
        self._pickle_dump(data, data_path)  # 数据持久化
        return np.array(data['inputs']), np.array(data['labels']), t2ix


class TestData(DataBase):
    def __init__(self, config):
        super(TestData, self).__init__(config)

    def load_vocab(self):
        word_to_index_path = os.path.join(self._output_path, 'word_to_index.pkl')
        label_to_index_path = os.path.join(self._output_path, 'label_to_index.pkl')
        with open(word_to_index_path, 'rb') as fr:
            word_to_index = pickle.load(fr)
        with open(label_to_index_path, 'rb') as fr:
            label_to_index = pickle.load(fr)
        return word_to_index, label_to_index

    def gen_test_data(self, data_path):
        """构造可导入模型的数据集"""
        inputs, labels = self.read_data(data_path)
        inputs = self.remove_stopwords_and_low_frequent(inputs)
        w2ix, t2ix = self.load_vocab()
        inputs = self.trans_w2ix(inputs, w2ix)
        labels = self.trans_t2ix_with_onehot(labels, t2ix)
        inputs = self.padding(inputs)

        data_path = os.path.join(self._output_path, 'test_data.pkl')
        data = {'inputs': inputs, 'labels': labels}
        self._pickle_dump(data, data_path)  # 数据持久化
        return np.array(data['inputs']), np.array(data['labels']), t2ix


if __name__ == '__main__':
    from config import Config

    cfg = Config()
    dataloader = TrainData(cfg)
    inputs, labels, t2ix = dataloader.gen_train_data(os.path.join(cfg.BASE_DIR, cfg.train_data_path))
    for d in dataloader.next_batch(inputs, labels, 32):
        print(d)
