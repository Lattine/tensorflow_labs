# -*- coding: utf-8 -*-

# @Time    : 2019/7/29
# @Author  : Lattine

# ======================
import argparse
import math
import time

import tensorflow as tf
import numpy as np

from data_input import DataLoader


def train(args):
    dataloader = DataLoader(args.data_dir, args.batch_size, args.win_size)
    args.vocab_size = dataloader.vocab_size

    graph = tf.Graph()
    with graph.as_default() as g:
        input_data = tf.placeholder(tf.int64, [args.batch_size, args.win_size])
        targets = tf.placeholder(tf.int64, [args.batch_size, 1])

        with tf.variable_scope('embedding'):
            embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim], -1.0, 1.0))
            embeddings = tf.nn.l2_normalize(embeddings, 1)

        with tf.variable_scope('weight'):
            weight_h = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim, args.hidden_num], stddev=1.0 / math.sqrt(args.hidden_num)))
            softmat_w = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim, args.vocab_size], stddev=1.0 / math.sqrt(args.win_size * args.word_dim)))
            softmat_u = tf.Variable(tf.truncated_normal([args.hidden_num, args.vocab_size], stddev=1.0 / math.sqrt(args.hidden_num)))
            b_1 = tf.Variable(tf.random_normal([args.hidden_num]))
            b_2 = tf.Variable(tf.random_normal([args.vocab_size]))

        def infer_output(input_data):
            """
                hidden = tanh(x*H + b1)
                output = softmax(x*W + hidden*U + b2)
            """
            input_data_emb = tf.nn.embedding_lookup(embeddings, input_data)
            input_data_emb = tf.reshape(input_data_emb, [-1, args.win_size * args.word_dim])
            hidden = tf.tanh(tf.matmul(input_data_emb, weight_h)) + b_1
            hidden_output = tf.matmul(hidden, softmat_u) + tf.matmul(input_data_emb, softmat_w) + b_2
            output = tf.nn.softmax(hidden_output)
            return output

        outputs = infer_output(input_data)
        one_hot_targets = tf.one_hot(tf.squeeze(targets), args.vocab_size, 1.0, 0.0)
        loss = -tf.reduce_mean(tf.reduce_sum(tf.log(outputs) * one_hot_targets, 1))
        optimizer = tf.train.AdagradOptimizer(0.1)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -args.grad_clip, args.grad_clip), var) for grad, var in gvs]
        optimizer = optimizer.apply_gradients(capped_gvs)

        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / embeddings_norm

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(args.num_epochs):
            dataloader.reset_batch_pointer()
            for b in range(dataloader.num_batches):
                start = time.time()
                x, y = dataloader.next_batch()
                feed = {input_data: x, targets: y}
                train_loss, _ = sess.run([loss, optimizer], feed)
                end = time.time()
                print("{}/{} (epoch {}), train loss:{:.3f}, time/batch:{:.3f}".format(b, dataloader.num_batches, e, train_loss, end - start))
            np.save(args.save_path + 'nnlm.zh', normalized_embeddings.eval())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r'data_set/', help=r'data set directory')
    parser.add_argument("--batch_size", type=int, default=128, help=r'batch size')
    parser.add_argument("--win_size", type=int, default=5, help=r'context window size')
    parser.add_argument("--hidden_num", type=int, default=64, help=r'dim of hidden layers')
    parser.add_argument("--word_dim", type=int, default=50, help=r'dim of word embedding')
    parser.add_argument("--num_epochs", type=int, default=10, help=r'number of epochs')
    parser.add_argument("--grad_clip", type=float, default=5., help=r'clip gradients threshold')
    parser.add_argument("--save_path", type=str, default=r'ckpt/', help=r'model save path')
    args = parser.parse_args()
    train(args=args)


if __name__ == '__main__':
    main()
