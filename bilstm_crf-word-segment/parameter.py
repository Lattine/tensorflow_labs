# -*- coding: UTF-8 -*-
# @Auther: Lattine
# @Date: 2019-05-25

class Parameters:
    # dataset
    train_data = './data/train.txt'
    test_data = './data/test.txt'
    val_data = './data/val.txt'
    dict_path = './data/word2dict.pkl' # 字典，文字到ID的映射

    # log path
    tensorboard_path = './tensorboard/logs'
    checkpoint_path = './ckpt'

    # train params
    epochs = 10

    batch_size = 64
    hidden_dim = 128
    embedding_size = 100
    vocab_size = 5000
    num_tags = 4
    
    learning_rate = 1e-4
    clip = 5.0
    keep_prob = 1.0