# -*- coding: utf-8 -*-

# @Time    : 2019/7/31
# @Author  : Lattine

import os


# ======================
class TextCNNConfig:
    BASE_DIR = os.path.abspath(os.path.dirname(os.getcwd()))
    model_name = "text_cnn"

    # Dataset
    train_data = "data/imdb/train_data.txt"
    eval_data = "data/imdb/eval_data.txt"
    test_data = "data/imdb/eval_data.txt"
    output_path = "data/outputs/imdb/text_cnn"
    word_vectors_path = None
    stopwords = "data/stopwords_en.txt"
    sequence_length = 500  # 这个可以自己设置，根据不同的语料调整
    embedding_size = 200
    vocab_size = 10000

    # Train & Eval
    num_classes = 1  # 二分类设置为1, 多分类设置为类别的数目
    epochs = 10
    batch_size = 50
    optimization = 'adam'
    learning_rate = 1e-3
    clip_grad = 5.0
    keep_prob = 0.5
    num_filters = 100
    filter_sizes = [3, 4, 5]
    l2_reg_lambda = 0.0
    ckeckpoint_every = 1
    ckpt_model_path = "ckpt/imdb/text_cnn"  # 模型保存路径
    summary_path = "ckpt/logs"  # 训练日志
