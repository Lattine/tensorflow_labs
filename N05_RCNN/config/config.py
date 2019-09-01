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
    embedding_dim = 200

    # Model
    model_output_size = 128
    hidden_sizes = [128]
    optimization = 'adam'
    l2_reg_lambda = 0.0
    learning_rate = 1e-3
    clip_grad = 5.0
    keep_prob = 0.5

    # Train
    epochs = 10
    batch_size = 128
    eval_every = 100
    ckpt_model_path = "ckpt/imdb/att_bilstm"  # 模型保存路径
    summary_path = "ckpt/logs"  # 训练日志
