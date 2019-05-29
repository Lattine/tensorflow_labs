# -*- coding: UTF-8 -*-
# @Auther: Lattine
# @Date: 2019-05-25

import os
import tensorflow as tf 
from parameter import Parameters as pm
from model import TextCNN
from data_processor import DataProcessor

class Trainer:
    def __init__(self, model, pm=pm):
        self.pm = pm
        self.model = model
        self._check_path()  # 检查tensorboard和checkpoint目录
        self.model_path = os.path.join(self.pm.checkpoint_path, self.model.name)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self._restore_model()

        tf.summary.scalar('loss', self.model.loss)
        tf.summary.scalar('accuracy', self.model.acc)
        self.merged_summary = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(pm.tensorboard_path)
        self.writer.add_graph(self.sess.graph)
    
    def train(self, dp):
        train_content, train_label = dp.sequence2id(self.pm.data_train)
        best_acc_val = 0.0
        cstep = 0
        early_stoped = False

        for epoch in range(self.pm.epochs):
            if early_stoped: break
            train_batch = dp.batch_iter(train_content, train_label, self.pm.batch_size)

            for x_batch, y_batch in train_batch:
                x_batch, y_batch = dp.process(x_batch, y_batch, pm.seq_length)
                feed_dict = self.model.feed_data(x_batch, y_batch, self.pm.keep_prob)

                _, gstep = self.sess.run([self.model.optim, self.model.global_step], feed_dict=feed_dict)

                if gstep % self.pm.summary_per_step == 0:
                    ms = self.sess.run(self.merged_summary, feed_dict=feed_dict)
                    self.writer.add_summary(ms, gstep)
                if gstep % self.pm.print_per_step == 0:
                    feed_dict = self.model.feed_data(x_batch, y_batch, 1.0)
                    loss_train, acc_train = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
                    loss_val, acc_val = self._evaluate(dp)
                    print(f"Epoch: {epoch}, Step: {gstep}, Train loss: {loss_train}, Train accuracy: {acc_train}, Val loss: {loss_val}, Val accuracy: {acc_val}")

                    if acc_val > best_acc_val:
                        cstep = 0
                        best_acc_val = acc_val
                        self.saver.save(sess=self.sess, save_path=self.model_path)

                if cstep > pm.early_stop_needed:
                    early_stoped = True
                    break

                cstep += 1
    
    def _evaluate(self, dp):
        val_content, val_label = dp.sequence2id(self.pm.data_val)
        length = len(val_content)
        val_batch = dp.batch_iter(val_content, val_label, 10*self.pm.batch_size)
        losses, acces = [], []

        for x, y in val_batch:
            x, y = dp.process(x, y, self.pm.seq_length)
            feed_dict = self.model.feed_data(x, y, 1.0)
            loss_val, acc_val = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            losses.append(loss_val)
            acces.append(acc_val)
        
        return sum(losses)/len(losses), sum(acces)/len(acces)
    
    def _restore_model(self):
        # 尝试加载历史最后一个模型
        files = os.listdir(self.pm.checkpoint_path)
        if len(files) > 3:
            self.sess.run(tf.global_variables_initializer())
            save_path = tf.train.latest_checkpoint(pm.checkpoint_path)
            self.saver.restore(sess=self.sess, save_path=save_path)

    def _check_path(self):
        if not os.path.exists(pm.tensorboard_path):
            os.makedirs(pm.tensorboard_path)
        if not os.path.exists(pm.checkpoint_path):
            os.makedirs(pm.checkpoint_path)

if __name__ == "__main__":
    dp = DataProcessor(pm.data_train, pm.dict_words, pm.dict_tags)
    model = TextCNN()
    t = Trainer(model=model)
    t.train(dp)