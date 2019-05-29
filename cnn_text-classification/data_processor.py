# -*- coding: UTF-8 -*-
# @Auther: Lattine
# @Date: 2019-05-28

import os
import pickle
import numpy as np
from parameter import Parameters as pm
import tensorflow.contrib.keras as kr

class DataProcessor(object):
    def __init__(self, corpus_path, dict_words, dict_tags, force_build=False):
        self.dict_words = dict_words
        self.dict_tags = dict_tags
        if force_build or (not os.path.exists(self.dict_words)):
            self._dict(corpus_path)

    def sequence2id(self, path):
        content2id, label2id = [], []
        _, _, contents, labels = self._read_file(path)
        with open(self.dict_words, 'rb') as fr:
            w2ix = pickle.load(fr)
        with open(self.dict_tags, 'rb') as fr:
            t2ix = pickle.load(fr)
        for i in range(len(contents)):
            content2id.append([w2ix.get(x, pm.UNK) for x in contents[i]])
        for i in range(len(labels)):
            label2id.append([t2ix.get(x) for x in labels[i]])
        return content2id, label2id
    
    def batch_iter(self, x, y, batch_size):
        length = len(x)
        x = np.array(x)
        y = np.array(y)
        num_batch = length // batch_size
        ixs = np.random.permutation(length)
        x_shuffle = x[ixs]
        y_shuffle = y[ixs]
        for i in range(num_batch):
            start = i*batch_size
            end = min((i+1)*batch_size, length)
            yield x_shuffle[start:end], y_shuffle[start:end]
        
    def process(self, x, y, max_length):
        x_padded = kr.preprocessing.sequence.pad_sequences(x, max_length)
        y_padded = kr.utils.to_categorical(y, num_classes=pm.num_tags)
        return x_padded, y_padded


    def _dict(self, corpus_path):
        words, tags, _, _ = self._read_file(corpus_path)
        chs = set(words)
        key_dict = {}
        key_dict['<PAD>'] = pm.PAD
        key_dict['<UNK>'] = pm.UNK
        it = 2
        for w in chs:
            key_dict[w] = it
            it += 1
        tags = set(tags)
        tag2ix = {t:i for i, t in enumerate(tags)}
        ix2tag = {i:t for i, t in enumerate(tags)}

        with open(self.dict_words, 'wb') as fw:
            pickle.dump(key_dict, fw)
        with open(self.dict_tags, 'wb') as fw:
            pickle.dump(tag2ix, fw)
            pickle.dump(ix2tag, fw)
    
    def _read_file(self, path):
        words, tags, content, label = [], [], [], []
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                word_list = self._get_word(line)
                label_list = self._get_label(line)
                words.extend(word_list)
                tags.extend(label_list)
                content.append(word_list)
                label.append(label_list)
        return words, tags, content, label
    
    def _get_word(self, line):
        word_list = []
        line = line.split("\t")
        for w in line[1]:
            word_list.append(w)
        return word_list
    
    def _get_label(self, line):
        label_list = []
        line = line.split("\t")
        label_list.append(line[0])
        return label_list
    

if __name__ == "__main__":
    p = DataProcessor(pm.data_train, pm.dict_words, pm.dict_tags, force_build=False)
    content, label = p.sequence2id(pm.data_val)
    batchs = p.batch_iter(content, label, 64)
    for x, y in batchs:
        x, y = p.process(x, y, 1000)
        print(x.shape, y.shape)
        # print(x[0])
        print(y[0])
        # break