# -*- coding: utf-8 -*-

# @Time    : 2019/7/31
# @Author  : Lattine

# ======================
from sklearn.metrics import roc_auc_score


def accuracy(pred_y, true_y):
    """计算二分类和分类的准确率"""
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc


def get_binary_metrics(pred_y, true_y):
    """二分类指标"""
    acc = accuracy(pred_y, true_y)
    return acc


def get_multi_metrics(pred_y, true_y):
    """二分类指标"""
    acc = accuracy(pred_y, true_y)
    return acc
