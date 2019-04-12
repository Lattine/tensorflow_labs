import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 下载数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


# 使用卷积神经网络来处理图片，核心部分就是卷积和池化

# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
# 这个方法是 TensorFlow 实现卷积常用的方法，也是搭建卷积神经网络的核心方法，参数介绍如下：
# input，指需要做卷积的输入图像，它要求是一个 Tensor，具有 [batch_size, in_height, in_width, in_channels] 这样的 shape，
# 具体含义是 [batch_size 的图片数量, 图片高度, 图片宽度, 输入图像通道数]，注意这是一个 4 维的 Tensor，要求类型为 float32 和 float64 其中之一。
# filter，相当于 CNN 中的卷积核，它要求是一个 Tensor，具有 [filter_height, filter_width, in_channels, out_channels] 这样的shape，
# 具体含义是 [卷积核的高度，卷积核的宽度，输入图像通道数，输出通道数（即卷积核个数）]，要求类型与参数 input 相同，有一个地方需要注意，第三维 in_channels，就是参数 input 的第四维。
# strides，卷积时在图像每一维的步长，这是一个一维的向量，长度 4，具有 [stride_batch_size, stride_in_height, stride_in_width, stride_in_channels] 这样的 shape，
# 第一个元素代表在一个样本的特征图上移动，第二三个元素代表在特征图上的高、宽上移动，第四个元素代表在通道上移动。
# padding，string 类型的量，只能是 SAME、VALID 其中之一，这个值决定了不同的卷积方式。 SAME=[n_out = ceil(n_in/s)]， padding=[n_out=ceil(n_in-f+1)/s]
# use_cudnn_on_gpu，布尔类型，是否使用 cudnn 加速，默认为true。
# 返回的结果是 [batch_size, out_height, out_width, out_channels] 维度的结果。

# tf.nn.max_pool(value, ksize, strides, padding, name=None)
# 是CNN当中的最大值池化操作，其实用法和卷积很类似。参数介绍如下：
# value，需要池化的输入，一般池化层接在卷积层后面，所以输入通常是 feature map，依然是 [batch_size, height, width, channels] 这样的shape。
# ksize，池化窗口的大小，取一个四维向量，一般是 [batch_size, height, width, channels]，因为不想在 batch 和 channels 上做池化，所以这两个维度设为了1。
# strides，和卷积类似，窗口在每一个维度上滑动的步长，一般也是 [stride_batch_size, stride_height, stride_width, stride_channels]。
# padding，和卷积类似，可以取 VALID、SAME，返回一个 Tensor，类型不变，shape 仍然是 [batch_size, height, width, channels] 这种形式。


def conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME"):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding)


def max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"):
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)


# 预定义
x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
y_label = tf.placeholder(tf.float32, shape=[None, 10])

# conv1
w_conv1 = weight([5, 5, 1, 32])
b_conv1 = bias([32])
x_reshape = tf.reshape(x, [-1, 28, 28, 1]) # 把 x 变为4维张量
h_conv1 = tf.nn.relu(conv2d(x_reshape, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)  # [-1, 14, 14, 32]

# conv2
w_conv2 = weight([5, 5, 32, 64])
b_conv2 = bias([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)  # [-1, 7, 7, 64]

# fc
w_fc1 = weight([7 * 7 * 64, 1024])
b_fc1 = bias([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
# TensorFlow 的 tf.nn.dropout 操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的 scale，所以用 dropout 的时候可以不用考虑 scale。

# output
w_fc2 = weight([1024, 10])
b_fc2 = bias([10])
y_ = tf.nn.softmax(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)

# loss
cross_entropy = -tf.reduce_sum(y_label * tf.log(y_))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# prediction
correct_prediction = tf.equal(tf.argmax(y_label, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# running train
# run graph
batch_size = 100
total_steps = 10000
steps_per_test = 100
dropout_keep_prob = 0.5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(total_steps + 1):
        bx, by = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: bx, y_label: by, keep_prob: dropout_keep_prob})
        if step % steps_per_test == 0:
            print(step, sess.run(accuracy, feed_dict={x: bx, y_label: by, keep_prob: 1.0}))

# final test
print("test accuracy : ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels, keep_prob: 1.0}))
