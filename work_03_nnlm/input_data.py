import os
import pickle
from collections import Counter

import numpy as np


class TextLoader:

    def __init__(self, data_dir, batch_size, seq_length, mini_frq=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, "input.zh.txt")
        vocab_file = os.path.join(data_dir, "vocab.zh.pkl")

        self._preprocess(input_file, vocab_file)
        self._create_batches()
        self.reset_pointer()

    def _preprocess(self, input_file, vocab_file):
        with open(input_file, "r", encoding="utf8") as fin:
            lines = fin.readlines()
            lines = [line.strip().split() for line in lines]

        self.vocab, self.words = self._build_vocab(lines)
        self.vocab_size = len(self.words)
        print(f"words num: {self.vocab_size}")

        with open(vocab_file, "wb") as fout:
            pickle.dump(self.words, fout)

        raw_data = [[0] * self.seq_length + [self.vocab.get(w, 1) for w in line] + [2] * self.seq_length for line in lines]
        self.raw_data = raw_data
        print(raw_data[0])

    def _build_vocab(self, sentences):
        counts = Counter()
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            counts.update(sent)
        words = ["<SOS>", "<UNK>", "<END>"] + [x[0] for x in counts.most_common() if [x[1] >= self.mini_frq]]
        vocab = {w: i for i, w in enumerate(words)}
        return [vocab, words]

    def _create_batches(self):
        xdata, ydata = [], []
        for row in self.raw_data:
            for ix in range(self.seq_length, len(row)):
                xdata.append(row[ix - self.seq_length:ix])
                ydata.append([row[ix]])
        self.num_batches = len(xdata) // self.batch_size
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.array(xdata[:self.num_batches * self.batch_size])
        ydata = np.array(ydata[:self.num_batches * self.batch_size])

        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_pointer(self):
        self.pointer = 0


if __name__ == "__main__":
    basedir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(basedir, "data")
    test = TextLoader(data_dir, 32, 10)
    for i in range(1):
        print(test.next_batch)
