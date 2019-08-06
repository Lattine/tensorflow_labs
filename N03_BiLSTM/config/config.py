# -*- coding: utf-8 -*-

# @Time    : 2019/8/5
# @Author  : Lattine

# ======================
import os


class BiLSTMConfig:
    BASE_DIR = os.path.abspath(os.path.dirname(os.getcwd()))
    model_name = "bilstm"

    # Dataset
    train_data_path = "data/imdb/train_data.txt"
    eval_data_path = "data/imdb/eval_data.txt"
    test_data_path = "data/imdb/eval_data.txt"
    output_path = "data/outputs/imdb/" + model_name
    stopwords_path = "data/stopwords_en.txt"
    word_vectors_path = "data/w2v_en.bin"
    vocab_size = 10000
    sequence_length = 200
    eval_every = 100

    # Model
    num_classes = 2  # 几分类
    embedding_dim = 200  # 词向量维度
    hidden_sizes = [256, 256]  # 隐藏层大小
    optimization = 'adam'
    l2_reg_lambda = 0.0
    learning_rate = 1e-3
    clip_grad = 5.0
    keep_prob = 0.5

    # Train
    epochs = 100
    batch_size = 128
    ckpt_model_path = "ckpt/imdb/" + model_name  # 模型保存路径
    summary_path = "ckpt/logs"  # 训练日志
