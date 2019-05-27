# -*- coding: UTF-8 -*-
# @Auther: Lattine
# @Date: 2019-05-25

import os
import re
import pickle
import numpy as np 
import tensorflow.contrib.keras as kr

state_list = {'B':0, 'M':1, 'E':2, 'S':3}
PAD = 0
UNK = 1


class Dataset:
    def __init__(self, corpus_path, w2id_path, force_build=False):
        self.w2id_path = w2id_path
        if force_build or (not os.path.exists(w2id_path)):
            self._word_dict(corpus_path)

    def sequence2id(self, path):
        # 将文字与标签，转换为ID
        content2id, label2id = [], []
        _, content, label = self.read_file(path)
        with open(self.w2id_path, 'rb') as fin:
            w2ix = pickle.load(fin)
        for i in range(len(label)):
            label2id.append([state_list[x] for x in label[i]])
        for j in range(len(content)):
            content2id.append([w2ix.get(x, UNK) for x in content[j]])
        return content2id, label2id
    
    def batch_iter(self, x, y, batch_size):
        length = len(x)
        x = np.array(x)
        y = np.array(y)
        num_batch = length // batch_size
        indices = np.random.permutation(length)
        x_shuffle = x[indices]
        y_shuffle = y[indices]
        for i in range(num_batch):
            start = i * batch_size
            end = min((i+1)*batch_size, length)
            yield x_shuffle[start:end], y_shuffle[start:end]
    
    def process(self, batch):
        # 计算一个batch中最长的句子，然后对所有句子做padding
        seq_len = []
        max_len = max(map(lambda x: len(x), batch))
        for i in range(len(batch)):
            seq_len.append(len(batch[i]))
        
        padded_batch = kr.preprocessing.sequence.pad_sequences(batch, max_len, padding='post', truncating='post')
        return padded_batch, seq_len
    
    def read_file(self, path):
        word, content, label = [], [], []
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                word_list = self.get_word(line)
                label_list = self.get_label(line)
                if len(word_list) != len(label_list):
                    print(f'error with {line} - {word_list} - {label_list}')
                    continue
                word.extend(word_list)
                content.append(word_list)
                label.append(label_list)
        return word, content, label
    
    def get_word(self, line):
        word_list = []
        sentence = re.sub(r'[ ]+', '', line)
        for w in sentence:
            word_list.append(w)
        return word_list
    
    def get_label(self, line):
        label_list = []
        words = re.split(r'[ ]+', line)
        for i in range(len(words)):
            if len(words[i]) == 1:
                label_list.append('S')
            elif len(words[i]) == 2:
                label_list.append('B')
                label_list.append('E')
            else:
                M_num = len(words[i]) - 2
                label_list.append('B')
                label_list.extend('M'*M_num)
                label_list.append('E')
        return label_list

    def _word_dict(self, corpus_path):
        words, _, _ = self.read_file(corpus_path)
        words = set(words)
        key_dict = {}
        key_dict['<PAD>'] = PAD
        key_dict['<UNK>'] = UNK
        ct = 2
        for w in words:
            key_dict[w] = ct
            ct += 1
        with open(self.w2id_path, 'wb') as fout:
            pickle.dump(key_dict, fout)

if __name__ == "__main__":
    batch_size = 64
    p = Dataset('./data/corpus.txt', './data/w2ix.pkl', force_build=True)
    content, label = p.sequence2id('./data/corpus.txt')
    num_batchs = len(content) // batch_size
    print(num_batchs)
    train_batch = p.batch_iter(content, label, batch_size)
    for x_batch, y_batch in train_batch:
        x_batch, x_length = p.process(x_batch)
        y_batch, y_length = p.process(y_batch)