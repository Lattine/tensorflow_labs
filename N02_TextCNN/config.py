# -*- coding: utf-8 -*-

# @Time    : 2019/7/30
# @Author  : Lattine

# ======================


class DataConfig:
    train_data = r'dataset/imdb'  # 训练数据
    output_path = r'dataset/imdb/outputs'  # 数据集处理的输出路径
    word_vectors_path = r'dataset'  # 词向量文件
    stopwords = r''  # 停用词文件
    sequence_length = 100  # 输入序列固定长度
    batch_size = 100  # 批次大小
    embedding_size = 100  # 词向量维度
    vocab_size = 10000  # 词汇表上限
