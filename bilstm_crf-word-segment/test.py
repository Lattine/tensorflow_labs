# -*- coding: UTF-8 -*-
# @Auther: Lattine
# @Date: 2019-05-27

# 预测
import os
import re
import pickle
import tensorflow as tf 
from parameter import Parameters as pm
from model import BiLSTM_CRF
from data_processor import Dataset, UNK

class Predictor:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.sess = tf.Session()

        self._restore_model()
    
    def _read_file(self, filename):
        content = []
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                words = self._get_words(line)
                content.append(words)
        return content
    
    def _get_words(self, line):
        word_list = []
        sentence = re.sub(r'[ ]+', '', line)
        for w in sentence:
            word_list.append(w)
        return word_list
    
    def sequence2id(self, filename):
        content2id = []
        content = self._read_file(filename)
        with open(pm.dict_path, 'rb') as fr:
            w2ix = pickle.load(fr)
        for i in range(len(content)):
            content2id.append([w2ix.get(x, UNK) for x in content[i]])
        return content2id
    
    def convert(self, sentence, label):
        word_cut = ''
        word_list = self._get_words(sentence)
        for i in range(len(label)):
            if label[i] == 2:
                word_cut += word_list[i]
                word_cut += ' '
            elif label[i] == 3:
                word_cut += ' '
                word_cut += word_list[i]
                word_cut += ' '
            else:
                word_cut += word_list[i]
        return word_cut
    
    def _restore_model(self):
        self.sess.run(tf.global_variables_initializer())
        save_path = tf.train.latest_checkpoint(pm.checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=save_path)

    def predict(self, evl_path):
        content = self.sequence2id(evl_path)
        labels = self.model.predict(self.sess, self.processor, content)
        return labels

if __name__ == "__main__":
    processor = Dataset(pm.train_data, pm.dict_path)
    model = BiLSTM_CRF()
    t = Predictor(model, processor)
    labels = t.predict(pm.val_data)

    with open(pm.val_data, 'r', encoding='utf-8') as fin:
        sentences = [line.strip() for line in fin]
    
    for i in range(len(sentences)):
        sent_cut = t.convert(sentences[i], labels[i])
        print(sentences[i])
        print(sent_cut)