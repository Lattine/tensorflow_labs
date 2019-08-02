# -*- coding: utf-8 -*-

# @Time    : 2019/7/31
# @Author  : Lattine

# ======================
from sklearn.metrics import roc_auc_score


def list_mean(items: list) -> float:
    """计算均值"""
    m = sum(items) / len(items) if len(items) > 0 else 0
    return m


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


def binary_auc(pred_y, true_y):
    auc = roc_auc_score(true_y, pred_y)
    return auc


def binary_precision(pred_y, true_y, positive=1):
    corr = 0
    pred_posi = 0
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_posi += 1
            if pred_y[i] == true_y[i]:
                corr += 1
    prec = corr / pred_posi if pred_posi > 0 else 0
    return prec


def binary_recall(pred_y, true_y, positive=1):
    corr = 0
    true_posi = 0
    for i in range(len(true_y)):
        if true_y[i] == positive:
            true_posi += 1
            if true_y[i] == pred_y[i]:
                corr += 1
    recall = corr / true_posi if true_posi > 0 else 0
    return recall


def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        fb = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        fb = 0
    return fb


"""
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0
    return f_b
"""


def get_binary_metrics(pred_y, true_y):
    """二分类指标"""
    acc = accuracy(pred_y, true_y)
    return acc


def get_multi_metrics(pred_y, true_y):
    """多分类指标"""
    acc = accuracy(pred_y, true_y)
    return acc
