# -*- coding: utf-8 -*-

# @Time    : 2019/8/19
# @Author  : Lattine

# ======================
import os


class RCNNConfig:
    # Base infos
    BASE_DIR = os.path.abspath(os.path.dirname(os.getcwd()))
    model_name = "rcnn"
    num_classes = 2  # 分类数量

    # Dataset
    train_data_path = "data/imdb/train_data.txt"
    eval_data_path = "data/imdb/eval_data.txt"
    test_data_path = "data/imdb/eval_data.txt"
    stopwords_path = "data/stopwords_en.txt"
    word_vectors_path = "data/w2v_en.bin"
    output_path = "data/outputs/imdb/" + model_name
    vocab_size = 10000
    sequence_length = 200
    embedding_size = 200
