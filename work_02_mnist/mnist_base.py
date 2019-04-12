import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 下载数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 以 placeholder 将现有的变量表示出来
x = tf.placeholder(tf.float32, [None, 28 * 28])
y_label = tf.placeholder(tf.float32, [None, 10])

# variable
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.nn.softmax(tf.matmul(x, w) + b)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y_), reduction_indices=[1]))
# 首先用 reduce_sum() 方法针对每一个维度进行求和，reduction_indices 是指定沿哪些维度进行求和。
# 然后调用 reduce_mean() 则求平均值，将一个向量中的所有元素求算平均值。

train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 测试模型

correct_prediction = tf.equal(tf.argmax(y_, axis=1), tf.argmax(y_label, axis=1))
# 首先让我们找出那些预测正确的标签。
# tf.argmax() 是一个非常有用的函数，它能给出某个 Tensor 对象在某一维上的其数据最大值所在的索引值。
# 由于标签向量是由 0,1 组成，因此最大值 1 所在的索引位置就是类别标签。
# 用 tf.equal() 方法来检测预测是否真实标签匹配（索引位置一样表示匹配）

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。


# run graph
batch_size = 100
total_steps = 10000
steps_per_test = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(total_steps + 1):
        bx, by = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: bx, y_label: by})
        if step % steps_per_test == 0:
            print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))
