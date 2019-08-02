# -*- coding: utf-8 -*-

# @Time    : 2019/8/2
# @Author  : Lattine

# ======================
import os

from config import TextCNNConfig
from predict import Predictor

config = TextCNNConfig()

with open(os.path.join(config.BASE_DIR, config.eval_data), 'r', encoding="utf8") as fr:
    inputs = []
    labels = []
    for line in fr:
        try:
            x, y = line.strip().split("<SEP>")
            inputs.append(x.strip().split(" "))
            labels.append(y.strip())
        except:
            print("Error with : ", line)

predictor = Predictor(config)

total = len(labels)

corr = 0
for sent, label in zip(inputs, labels):
    pred = predictor.predict(sent)
    if str(pred) == label: corr += 1

print(f"total: {total}, num class: {len(set(labels))}, correction: {corr/total}")
