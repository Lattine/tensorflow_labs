# -*- coding: UTF-8 -*-
# @Auther: Lattine
# @Date: 2019-05-25

import os
from sklearn import metrics
import tensorflow as tf 
from parameter import Parameters as pm
from model import TextCNN
from data_processor import DataProcessor

class Tester:
    def __init__(self, model, pm=pm):
        self.pm = pm
        self.model = model
        self.model_path = os.path.join(self.pm.checkpoint_path, self.model.name)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self._restore_model()
    
    def test(self, dp):
        test_content, test_label = dp.sequence2id(self.pm.data_test)

        test_batch = dp.batch_iter(test_content, test_label, 10*self.pm.batch_size)
        losses, acces = [], []

        for x_batch, y_batch in test_batch:
            x_batch, y_batch = dp.process(x_batch, y_batch, pm.seq_length)
            feed_dict = self.model.feed_data(x_batch, y_batch, 1.0)
            loss_test, acc_test = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            losses.append(loss_test)
            acces.append(acc_test)
            # y_pred = self.sess.run(self.model.y_pred, feed_dict=feed_dict)
            # print("Precision, Recall and F1-Score ...")
            # print(metrics.classification_report(y_batch, y_pred))
            # print("Confusion Matrix ...")
            # print(metrics.confusion_matrix(y_batch, y_pred))
        print(f"Test loss: {sum(losses)/len(losses)}, Test accuracy: {sum(acces)/len(acces)}")

    def _restore_model(self):
        # 尝试加载历史最后一个模型
        files = os.listdir(self.pm.checkpoint_path)
        if len(files) > 3:
            self.sess.run(tf.global_variables_initializer())
            save_path = tf.train.latest_checkpoint(pm.checkpoint_path)
            self.saver.restore(sess=self.sess, save_path=save_path)


if __name__ == "__main__":
    dp = DataProcessor(pm.data_train, pm.dict_words, pm.dict_tags)
    model = TextCNN()
    t = Tester(model=model)
    t.test(dp)