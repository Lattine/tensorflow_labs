import tensorflow as tf 
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
from tensorflow.examples.tutorials.mnist import input_data

# 下载数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Common
time_step = 28 # 时序长度，即每做一次预测需要输入多少步
input_size = 28 # 每个时刻的输入特征维度
num_units = 256 # 每个隐藏层的节点个数
num_layers = 3 # RNN 层数
category_num = 10 # 输出分类的类别数量，如果是回归预测的话应该是1
total_steps = 2000
steps_per_validate = 100
steps_per_test = 500
learning_rate = 1e-3

# pre_defined
x = tf.placeholder(tf.float32, [None, 28*28])
y_label = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, []) # TF-v1.0之后，这里需要[]，

# 需要将一张图分为多个 time_step 来输入，这样才能构建一个 RNN 序列，
# 所以这里直接将 time_step 设成 28，这样一来 input_size 就变为了 28，batch_size 不变，所以reshape 的结果是一个三维的矩阵[batch, time_step, input_size]

# STep 1. RNN 的输入 shape = [batch_size, timestep_size, input_size]
x_reshape = tf.reshape(x, [-1, time_step, input_size])


# Step 2. 定义一层LSTM cell，并包装Dropout layer，只要说明hidden_size,它会自动匹配输入的 X 的维度
def cell(num_units):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
    return DropoutWrapper(cell, output_keep_prob=keep_prob)

# Step 3. 通过定义的cell函数生成多层LSTM
cells = tf.nn.rnn_cell.MultiRNNCell([cell(num_units) for _ in range(num_layers)])
# 这里是循环一次新生成一个 LSTMCell，而不是直接使用乘法来扩展列表，因为会导致 LSTMCell 是同一个对象，导致构建完 MultiRNNCell 之后出现维度不匹配的问题。

# Step 4. 全零初始化state
h0 = cells.zero_state(batch_size, dtype=tf.float32)

# 步骤6：方法一，调用 dynamic_rnn() 来让构建好的网络运行起来
#  当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size] 
# 返回结果有两个，一个 output 是所有 time_step 的输出结果，赋值为 output，它是三维的，第一维长度等于 batch_size，第二维长度等于 time_step，第三维长度等于 num_units。
# 另一个 hs 是隐含状态，是元组形式，长度即 RNN 的层数 3，每一个元素都包含了 c 和 h，即 LSTM 的两个隐含状态。
output, hs = tf.nn.dynamic_rnn(cells, inputs=x_reshape, initial_state=h0)

# 获取最后一个time_step的结果
output = output[:, -1, :]

# 接下来做一次线性变换和Softmax输出结果
w = tf.Variable(tf.truncated_normal([num_units, category_num], stddev=0.1), dtype=tf.float32)
b = tf.Variable(tf.constant(0.1, shape=[category_num]), dtype=tf.float32)
y = tf.matmul(output, w) +b

# loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=y)

# 训练和评估
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_label, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# running
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(total_steps + 1):
        bx, by = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: bx, y_label: by, keep_prob:0.5, batch_size:bx.shape[0]})

        if step % steps_per_validate == 0:
            print("train", step, sess.run(accuracy, feed_dict={x:bx, y_label:by, keep_prob:1.0, batch_size:bx.shape[0]}))
        
        if step % steps_per_test == 0:
            tx, ty = mnist.test.images, mnist.test.labels
            print("test", step, sess.run(accuracy, feed_dict={x:tx, y_label:ty, keep_prob:1.0, batch_size:tx.shape[0]}))