# -*- coding: utf-8 -*-

# @Time    : 2019/7/31
# @Author  : Lattine

# ======================
import os
import json
import argparse

import tensorflow as tf

from config import TextCNNConfig
from data_helper import TrainData
from data_helper import TestData
from metrics import get_binary_metrics, get_multi_metrics, list_mean
from model import TextCNN


class Trainer:
    def __init__(self, config):
        self.config = config
        self.train_data_loader = None
        self.eval_data_loader = None

        # 加载数据集
        self.load_data()
        self.train_inputs, self.train_labels, label_to_idx = self.train_data_loader.gen_data()
        self.vocab_size = self.train_data_loader.vocab_size
        self.word_vectors = self.train_data_loader.word_vectors
        print(f"train data size: {len(self.train_labels)}")
        print(f"vocab size: {self.vocab_size}")
        self.label_list = [value for key, value in label_to_idx.items()]

        self.eval_inputs, self.eval_labels = self.eval_data_loader.gen_data()

        # 初始化模型
        self.model = TextCNN(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)

    def load_data(self):
        """加载数据集"""
        self.train_data_loader = TrainData(self.config)
        self.config.test_data = self.config.eval_data  # 使用验证集，进行训练过程中的测试
        self.eval_data_loader = TestData(self.config)

    def train(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())  # 初始化变量
            current_step = 0

            # 创建Train/Eval的summar路径和写入对象
            train_summary_path = os.path.join(self.config.BASE_DIR, self.config.summary_path + "/train")
            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            train_summary_writer = tf.summary.FileWriter(train_summary_path, sess.graph)
            eval_summary_path = os.path.join(self.config.BASE_DIR, self.config.summary_path + "/eval")
            if not os.path.exists(eval_summary_path):
                os.makedirs(eval_summary_path)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_path, sess.graph)

            # Train & Eval Process
            for epoch in range(self.config.epochs):
                print(f"----- Epoch {epoch + 1}/{self.config.epochs} -----")
                for batch in self.train_data_loader.next_batch(self.train_inputs, self.train_labels, self.config.batch_size):
                    summary, loss, predictions = self.model.train(sess, batch, self.config.keep_prob)
                    train_summary_writer.add_summary(summary)
                    if self.config.num_classes == 1:
                        acc = get_binary_metrics(pred_y=predictions.tolist(), true_y=batch['y'])
                        print("Train step: {}, acc: {:.3f}".format(current_step, acc))
                    elif self.config.num_classes > 1:
                        acc = get_multi_metrics(pred_y=predictions.tolist(), true_y=batch['y'])
                        print("Train step: {}, acc: {:.3f}".format(current_step, acc))

                    current_step += 1

                    if self.eval_data_loader and current_step % self.config.ckeckpoint_every == 0:
                        eval_losses = []
                        eval_accs = []
                        for eval_batch in self.eval_data_loader.next_batch(self.eval_inputs, self.eval_labels, self.config.batch_size):
                            eval_summary, eval_loss, eval_predictions = self.model.eval(sess, eval_batch)
                            eval_summary_writer.add_summary(eval_summary)
                            eval_losses.append(eval_loss)
                            if self.config.num_classes == 1:
                                acc = get_binary_metrics(pred_y=eval_predictions.tolist(), true_y=batch['y'])
                                eval_accs.append(acc)
                            elif self.config.num_classes > 1:
                                acc = get_multi_metrics(pred_y=eval_predictions.tolist(), true_y=batch['y'])
                                eval_accs.append(acc)
                        print(f"Eval \tloss: {list_mean(eval_losses)}, acc: {list_mean(eval_accs)}")

                        if self.config.ckpt_model_path:
                            save_path = os.path.join(self.config.BASE_DIR, self.config.ckpt_model_path)
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config.model_name)
                            self.model.saver.save(sess, model_save_path, global_step=current_step)


if __name__ == '__main__':
    config = TextCNNConfig()

    # 训练器
    trainer = Trainer(config)
    trainer.train()
