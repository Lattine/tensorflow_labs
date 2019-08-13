# -*- coding: utf-8 -*-

# @Time    : 2019/8/7
# @Author  : Lattine

# ======================
import os
import pickle

import numpy as np
import tensorflow as tf
from model import Model


class Predictor:
    def __init__(self, config):
        self.config = config
        self.output_path = os.path.join(self.config.BASE_DIR, self.config.output_path)

        self.w2ix, self.ix2t = self.load_vocab()  # 加载索引字典
        self.vocab_size = len(self.w2ix)
        self.sequence_length = self.config.sequence_length

        self.model = Model(self.config, self.vocab_size)
        self.load_graph()

    def load_vocab(self):
        with open(os.path.join(self.output_path, 'word_to_index.pkl'), 'rb') as fr:
            word_to_index = pickle.load(fr)

        with open(os.path.join(self.output_path, 'label_to_index.pkl'), 'rb') as fr:
            label_to_index = pickle.load(fr)
        index_to_label = {v: k for k, v in label_to_index.items()}
        return word_to_index, index_to_label

    def load_graph(self):
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(os.path.join(self.config.BASE_DIR, self.config.ckpt_model_path))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reloading model parameters..")
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("No such file: [{}]".format(self.config.ckpt_model_path))

    def sentence_to_ids(self, sentence):
        sentence_ids = [self.w2ix.get(token, self.w2ix.get("<UNK>")) for token in sentence]
        sentence_padded = [sentence_ids[:self.sequence_length] if len(sentence_ids) > self.sequence_length else sentence_ids + [0] * (self.sequence_length - len(sentence_ids))]
        return sentence_padded

    def predict(self, sentence):
        sentence_idx = self.sentence_to_ids(sentence)
        sentence_idx = np.array(sentence_idx, dtype='int64')
        prediction = self.model.predict(self.sess, sentence_idx).tolist()
        label = self.ix2t[prediction[0]]
        return label
