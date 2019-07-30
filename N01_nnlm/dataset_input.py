# -*- coding: utf-8 -*-

# @Time    : 2019/7/29
# @Author  : Lattine

# ======================
import os
import codecs
import pickle
from collections import Counter

import numpy as np


class DataLoader:
    def __init__(self, data_dir, batch_size, seq_length, mini_frq=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, 'input.zh.txt')
        vocab_file = os.path.join(data_dir, 'vocab.zh.pkl')

        self.preprocess(input_file, vocab_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file):
        with open(input_file, 'r', encoding="utf-8") as fr:
            lines = fr.readlines()
            if lines[0][:1] == codecs.BOM_UTF8:  # 某些文件开头会有特殊编码
                lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines]

        self.vocab, self.words = self.build_vocab(lines)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as fw:
            pickle.dump(self.vocab, fw)

        raw_data = [[0] * self.seq_length + [self.vocab.get(w, 1) for w in line] + [2] * self.seq_length for line in lines]
        self.raw_data = raw_data

    def build_vocab(self, sentences):
        word_counts = Counter()
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            word_counts.update(sent)

        vocabulary_inv = ["<START>", "<UNK>", "<END>"] + [c[0] for c in word_counts.most_common() if c[1] >= self.mini_frq]
        vocabulary = {w: i for i, w in enumerate(vocabulary_inv)}
        return vocabulary, vocabulary_inv

    def create_batches(self):
        xdata, ydata = list(), list()
        for row in self.raw_data:
            for ind in range(self.seq_length, len(row)):
                xdata.append(row[ind - self.seq_length:ind])
                ydata.append([row[ind]])
        self.num_batches = len(xdata) // self.batch_size
        if self.num_batches == 0:
            assert False, "Not enough data."
        xdata = np.array(xdata[:self.num_batches * self.batch_size])
        ydata = np.array(ydata[:self.num_batches * self.batch_size])
        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
