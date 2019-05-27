# -*- coding: UTF-8 -*-
# @Auther: Lattine
# @Date: 2019-05-25

import os
import re
import tensorflow as tf 
from parameter import Parameters as pm
from model import BiLSTM_CRF
from data_processor import Dataset

class Trainer:
    def __init__(self, model, pm=pm):
        self.pm = pm
        self.model = model
        self._check_path()  # 检查tensorboard和checkpoint目录
        self.model_path = os.path.join(self.pm.checkpoint_path, 'cws.model')

        self.sess = tf.Session()

        tf.summary.scalar('loss', self.model.loss)
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(pm.tensorboard_path)
        self.writer.add_graph(self.sess.graph)
        self.saver = tf.train.Saver()


        self.sess.run(tf.global_variables_initializer())
        self._restore_model()
    
    def train(self, processor):
        train_content, train_label = processor.sequence2id(self.pm.train_data)
        test_content, test_label = processor.sequence2id(self.pm.test_data)

        for epoch in range(self.pm.epochs):
            print(f"Epoch : {epoch}")
            num_batchs = len(train_content) // self.pm.batch_size
            train_batch = processor.batch_iter(train_content, train_label, self.pm.batch_size)
            for x_batch, y_batch in train_batch:
                x_batch, x_length = processor.process(x_batch)
                y_batch, y_length = processor.process(y_batch)
                feed_dict = self.model.feed_data(x_batch, y_batch, x_length, self.pm.keep_prob)
                _, global_step, loss, train_summary = self.sess.run([self.model.optimizer, self.model.global_step, self.model.loss, self.merged_summary], feed_dict=feed_dict)
                if global_step % 100 == 0:
                    test_loss = self.model.test(self.sess, processor, test_content, test_label)
                    print(f'Epoch: {epoch}, global_step: {global_step}, train_loss: {loss}, test_loss: {test_loss}')

                if global_step % (num_batchs*2) == 0:
                    print(f"Epoch: {epoch}, global_step: {global_step}, saving model...")
                    self.saver.save(self.sess, self.model_path, global_step=global_step)
    
    def _restore_model(self):
        # 尝试加载历史最后一个模型
        files = os.listdir(self.pm.checkpoint_path)
        if len(files) > 3:
            self.sess.run(tf.global_variables_initializer())
            save_path = tf.train.latest_checkpoint(pm.checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(sess=self.sess, save_path=save_path)

    def _check_path(self):
        if not os.path.exists(pm.tensorboard_path):
            os.makedirs(pm.tensorboard_path)
        if not os.path.exists(pm.checkpoint_path):
            os.makedirs(pm.checkpoint_path)

if __name__ == "__main__":
    processor = Dataset(pm.train_data, pm.dict_path)
    model = BiLSTM_CRF()
    t = Trainer(model=model)
    t.train(processor)