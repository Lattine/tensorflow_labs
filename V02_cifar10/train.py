# -*- coding: utf-8 -*-

# @Time    : 2019/7/20
# @Author  : Lattine

# ======================
import tensorflow as tf
import cifar10
import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', r'./data_cifar10/cifar10_train', 'Directory for train data and checkpoint.')
tf.app.flags.DEFINE_integer('max_steps', 100000, 'Number of batches to run.')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_integer('log_frequency', 100, "Log frequency.")


def train():
    g = tf.Graph()
    with g.as_default() as g:
        global_step = tf.contrib.framework.get_or_create_global_step()
        images, labels = cifar10.distorted_inputs()
        logits = cifar10.inference(images)
        loss = cifar10.loss(logits, labels)

        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time
                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    print(f"{datetime.now()}: step {self._step}, loss = {loss_value}({int(examples_per_sec)} examples/sec;{int(sec_per_batch)} sec/batch )")

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps), tf.train.NanTensorHook(loss), _LoggerHook()],
                config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
