# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#ｙ为预测的值(书本定义)
y = tf.nn.softmax(tf.matmul(x, w) + b)
#y_为实际的值
y_ = tf.placeholder(tf.float32, [None, 10])
#tf.reduce_mean是对结果求平均值,reduce_sum是对结果求和,reduction_indices不知道用来干嘛
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#接下来是优化函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys})
	
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))
# 输出
# (55000, 784) (55000, 10)
# (10000, 784) (10000, 10)
# (5000, 784) (5000, 10)
# 2018-12-28 11:44:52.304474: W tensorflow/core/framework/allocator.cc:122] Allocation of 31360000 exceeds 10% of system memory.
# 0.9189
