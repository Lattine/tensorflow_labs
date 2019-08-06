# -*- coding: utf-8 -*-

# @Time    : 2019/8/5
# @Author  : Lattine

# ======================
import os
import tensorflow as tf
from config import BiLSTMConfig
from data_helper import TrainData, TestData
from model import BiLSTM


class Trainer:
    def __init__(self, config):
        self.config = config

        self.load_data()  # 加载数据集
        self.model = BiLSTM(self.config, self.vocab_size, self.word_vectors)  # 初始化模型

    def load_data(self):
        self.train_dataloader = TrainData(self.config)
        self.eval_dataloader = TestData(self.config)
        train_data_path = os.path.join(self.config.BASE_DIR, self.config.train_data_path)
        self.train_inputs, self.train_labels, self.t2ix = self.train_dataloader.gen_train_data(train_data_path)
        eval_data_path = os.path.join(self.config.BASE_DIR, self.config.eval_data_path)
        self.eval_inputs, self.eval_labels, _ = self.eval_dataloader.gen_test_data(eval_data_path)
        self.vocab_size = self.train_dataloader.vocab_size
        self.word_vectors = self.train_dataloader.word_vectors

    def train(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())  # 初始化计算图变量
            current_step = 0

            # 创建Train/Eval的summar路径和写入对象
            train_summary_path = os.path.join(self.config.BASE_DIR, self.config.summary_path + "/train")
            eval_summary_path = os.path.join(self.config.BASE_DIR, self.config.summary_path + "/eval")
            self._check_directory(train_summary_path)
            self._check_directory(eval_summary_path)
            train_summary_writer = tf.summary.FileWriter(train_summary_path, sess.graph)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_path, sess.graph)

            # Train & Eval Process
            for epoch in range(self.config.epochs):
                print(f"----- Epoch {epoch + 1}/{self.config.epochs} -----")
                for batch in self.train_dataloader.next_batch(self.train_inputs, self.train_labels, self.config.batch_size):
                    summary, loss, predictions = self.model.train(sess, batch, self.config.keep_prob)
                    accuracy = self.model.get_metrics(sess, batch)
                    train_summary_writer.add_summary(summary, current_step)
                    print(f"! Train epoch: {epoch}, step: {current_step}, train loss: {loss}, accuracy: {accuracy}")

                    current_step += 1
                    if self.eval_dataloader and current_step % self.config.eval_every == 0:
                        losses = []
                        acces = []
                        for eval_batch in self.eval_dataloader.next_batch(self.eval_inputs, self.eval_labels, self.config.batch_size):
                            eval_summary, eval_loss, eval_predictions = self.model.eval(sess, eval_batch)
                            eval_accuracy = self.model.get_metrics(sess, batch)
                            eval_summary_writer.add_summary(eval_summary, current_step)
                            losses.append(eval_loss)
                            acces.append(eval_accuracy)
                        print(f"! Eval epoch: {epoch}, step: {current_step}, eval loss: {sum(losses) / len(losses)}, accuracy: {sum(acces) / len(acces)}")

                        if self.config.ckpt_model_path:
                            save_path = os.path.join(self.config.BASE_DIR, self.config.ckpt_model_path)
                            self._check_directory(save_path)
                            model_save_path = os.path.join(save_path, self.config.model_name)
                            self.model.saver.save(sess, model_save_path, global_step=current_step)

    def _check_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == '__main__':
    config = BiLSTMConfig()

    # 训练器
    trainer = Trainer(config)
    trainer.train()
