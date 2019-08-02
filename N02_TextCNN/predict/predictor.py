# -*- coding: utf-8 -*-

# @Time    : 2019/8/2
# @Author  : Lattine

# ======================

import os
import pickle

import tensorflow as tf

from model import TextCNN


class Predictor:
    def __init__(self, config):
        self.config = config
        self.output_path = os.path.join(config.BASE_DIR, config.output_path)

        self.word_to_index, self.label_to_index = self.load_vocab()
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.vocab_size = len(self.word_to_index)
        self.word_vectors = None
        self.sequence_length = self.config.sequence_length

        self.model = TextCNN(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)

        self.load_graph()

    def load_vocab(self):
        with open(os.path.join(self.output_path, 'word_to_index.pkl'), 'rb') as fr:
            word_to_index = pickle.load(fr)

        with open(os.path.join(self.output_path, 'label_to_index.pkl'), 'rb') as fr:
            label_to_index = pickle.load(fr)

        return word_to_index, label_to_index

    def load_graph(self):
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(os.path.join(self.config.BASE_DIR, self.config.ckpt_model_path))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reloading model parameters..")
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("No such file: [{}]".format(self.config.ckpt_model_path))

    def sentence_to_ids(self, sentence):
        sentence_ids = [self.word_to_index.get(token, self.word_to_index.get("<UNK>")) for token in sentence]
        sentence_padded = [sentence_ids[:self.sequence_length] if len(sentence_ids) > self.sequence_length else sentence_ids + [0] * (self.sequence_length - len(sentence_ids))]
        return sentence_padded

    def predict(self, sentence):
        sentence_ids = self.sentence_to_ids(sentence)
        prediction = self.model.infer(self.sess, sentence_ids).tolist()
        label = self.index_to_label[prediction[0][0]]
        return label
