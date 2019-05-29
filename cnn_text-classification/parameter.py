# -*- coding: UTF-8 -*-
# @Auther: Lattine
# @Date: 2019-05-28

class Parameters(object):
    # dataset
    data_train = './data/cnews.train.txt'
    data_test = './data/cnews.test.txt'
    data_val = './data/cnews.val.txt'
    dict_words = './data/dict_words.pkl'
    dict_tags = './data/dict_tags.pkl'

    # Pre-defined Param
    PAD = 0
    UNK = 1

    # Model Params
    seq_length = 1000
    num_tags = 10
    vocab_size = 7000
    embedding_dim = 64
    hidden_dim = 128
    num_filters = 256
    kernel_size = 5
    clip = 5.0

    # Saver 
    tensorboard_path = './tensorboard/logs'
    checkpoint_path = './ckpt'
    summary_per_step = 10
    print_per_step = 100
    early_stop_needed = 10 * print_per_step 


    # Train
    learning_rate = 1e-3
    epochs = 10
    batch_size = 64
    keep_prob = 0.5

