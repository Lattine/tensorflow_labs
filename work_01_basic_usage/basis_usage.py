import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(2, 100))  # [2, 100], 代表三维点的（x, y）坐标
y_data = np.dot([0.3, 0.2], x_data) + 0.4  # [100]，代表 z 轴坐标
print(x_data)
print(y_data)

# 构造模型
# 拟合的过程实际上就是寻找 (x, y) 和 z 的关系, 即变量 x_data 和变量 y_data 的关系
# 首先将现有的变量来表示出来，用 placeholder() 方法声明，在运行的时候传递给它真实的数据，第一个参数是数据类型，第二个参数是形状
x = tf.placeholder(tf.float32, [2, 100])
y = tf.placeholder(tf.float32, [100])

# 即 z = w * (x, y) + b，所以拟合的过程实际上就是找 w 和 b 的过程，所以这里我们就首先像设变量一样来设两个变量 w 和 b
b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([2], -1.0, 1.0))
y_ = tf.matmul(tf.reshape(w, [1, 2]), x) + b  # matmul是TF中的矩阵乘法，类似np.dot，但是不能做向量与矩阵的乘法，所以要先将向量reshape成矩阵

# 损失函数
loss = tf.reduce_mean(tf.square(y - y_)) # 损失函数代表模型实际输出值和真实值之间的差距，目的就是来减小这个损失

optimizier = tf.train.GradientDescentOptimizer(0.5)
train = optimizier.minimize(loss)

# 运行模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        sess.run(train, feed_dict={x: x_data, y: y_data})
        if step % 10 == 0:
            print(step, sess.run(w), sess.run(b))