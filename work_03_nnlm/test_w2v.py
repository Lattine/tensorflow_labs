import pickle
import numpy as np
import os
import math

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data")
vocab_file = os.path.join(data_dir, "vocab.zh.pkl")

with open(vocab_file, "rb") as fin:
    vocab = pickle.load(fin, encoding="bytes")
word_emb = np.load("nnlm_word_embeddings.zh" + ".npy")
vocab = {v: k for k, v in enumerate(vocab)}
word1_ix = vocab["中国"]
word2_ix = vocab["美国"]

word1_emb = word_emb[word1_ix]
word2_emb = word_emb[word2_ix]


def consine_distance(vec1, vec2):
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot_product += a * b
        norm1 += a**2
        norm2 += b**2
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    else:
        return dot_product / ((norm1 * norm2)**0.5)


cos_sim = np.dot(word1_emb, word2_emb) / (np.linalg.norm(word1_emb) * np.linalg.norm(word2_emb))

# cosine distance = 1 - cosine similarity
cos_dist = 1.0 - cos_sim

print(consine_distance(word1_emb, word2_emb))
print(cos_dist)