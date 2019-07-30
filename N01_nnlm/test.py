# -*- coding: utf-8 -*-

# @Time    : 2019/7/30
# @Author  : Lattine

# ======================
import argparse
import os
import pickle

import numpy as np


def test(args):
    vocab_file = os.path.join(args.data_dir, 'vocab.zh.pkl')
    model_path = os.path.join(args.model_dir, 'w2v.zh.npy')
    with open(vocab_file, 'rb') as fr:
        vocab = pickle.load(fr)
    word_emb = np.load(model_path)
    word1_id = vocab['中国']
    word2_id = vocab['美国']
    word1_emb = word_emb[word1_id]
    word2_emb = word_emb[word2_id]

    def cosine_distance(vec1, vec2):
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for a, b in zip(vec1, vec2):
            dot_product += a * b
            norm1 += a ** 2
            norm2 += b ** 2
        if norm1 == 0.0 or norm2 == 0.0:
            return None
        else:
            return dot_product / ((norm1 * norm2) ** 0.5)

    print(cosine_distance(word1_emb, word2_emb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r'data_set', help=r'data directory')
    parser.add_argument("--model_dir", type=str, default=r'ckpt', help=r'data directory')
    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
