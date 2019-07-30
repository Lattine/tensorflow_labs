# -*- coding: utf-8 -*-

# @Time    : 2019/7/30
# @Author  : Lattine

# ======================
import os
import pickle

import numpy as np

from .base import TestDataBase


class TestData(TestDataBase):
    def __init__(self, config):
        super(TestData, self).__init__(config)

        self._test_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config.test_data)
        self._output_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd(), config.output_path)))
        self._stopwords_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config.stopwords) if config.stopwords else None
        self._sequence_length = config.sequence_length

    def read_data(self):
        """读取数据"""
        inputs, labels = [], []
        with open(self._test_data_path, 'r', encoding='utf8') as fr:
            for line in fr:
                try:
                    text, label = line.strip().split("<SEP>")
                    inputs.append(text.strip().split(" "))
                    labels.append(label)
                except:
                    continue
        return inputs, labels

    def remove_stopwords(self, inputs):
        """过滤停用词"""
        if self._stopwords_path:
            with open(self._stopwords_path, 'r', encoding='utf8') as fr:
                stopwords = [line.strip() for line in fr]
            inputs = [[w for w in sentence if w not in stopwords] for sentence in inputs]
        return inputs

    def load_vocab(self):
        """加载词汇-索引字典"""
        word_to_index_path = os.path.join(self._output_path, 'word_to_index.pkl')
        label_to_index_path = os.path.join(self._output_path, 'label_to_index.pkl')
        with open(word_to_index_path, 'rb') as fr:
            word_to_index = pickle.load(fr)
        with open(label_to_index_path, 'rb') as fr:
            label_to_index = pickle.load(fr)

        return word_to_index, label_to_index

    @staticmethod
    def trans_w2ix(inputs, w2ix):
        """文本转为索引"""
        inputs_idx = [[w2ix.get(w, w2ix.get("<UNK>")) for w in sentence] for sentence in inputs]
        return inputs_idx

    @staticmethod
    def trans_t2ix(labels, t2ix):
        """标签转索引"""
        labels_idx = [t2ix.get(label) for label in labels]
        return labels_idx

    def padding(self, inputs, sequence_length):
        """对序列截断/填充"""
        new_inputs = [sentence[:sequence_length] if len(sentence) > sequence_length else sentence + [0] * (sequence_length - len(sentence)) for sentence in inputs]
        return new_inputs

    def gen_data(self):
        """生成可导入模型的数据"""
        eval_data_path = os.path.join(self._output_path, 'eval_data.pkl')
        # 如果存在，则直接加载
        if os.path.exists(eval_data_path):
            print(f"load existed eval data")
            with open(eval_data_path, 'rb') as fr:
                eval_data = pickle.load(fr)
            return np.array(eval_data['inputs_idx']), eval_data['labels_idx']

        # --------- 原始处理流程 ----------
        # 1.读取原始数据
        inputs, labels = self.read_data()
        print("read data finished")

        # 2.过滤停用词
        inputs = self.remove_stopwords(inputs)
        print('remove stopwords finished')

        # 3.加载词汇-索引字典
        word_to_index, label_to_index = self.load_vocab()
        print('load vocab finished')

        # 4.文本转索引
        inputs_idx = self.trans_w2ix(inputs, word_to_index)
        print('trans w2ix finished')
        # 5. 对序列做PADDING
        inputs_idx = self.padding(inputs, self._sequence_length)
        print('inputs padding finished')

        # 6.标签转索引
        labels_idx = self.trans_t2ix(labels, label_to_index)
        print('trans t2ix finished')

        # 构建Eval数据字典，保持为文件
        eval_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
        with open(eval_data_path, 'wb') as fw:
            pickle.dump(eval_data, fw)

        return np.array(inputs_idx), np.array(labels_idx)


def next_batch(self, x, y, batch_size):
    """生成Batch数据"""
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    num_batches = len(x) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_x = np.array(x[start:end], dtype='int64')
        batch_y = np.array(y[start:end], dtype='float32')

        yield batch_x, batch_y
