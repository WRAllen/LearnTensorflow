# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

#创建权重的随机噪点
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#二维卷积函数和卷积模板移动的步长（都是１表示不会遗漏任何一个点）
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#max_pool最大池化和移动的长（便是横竖两个方向以２为步长）
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
# 定义输入和编写输入形状将1D转回原来的2D                     
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])
# 第一次卷积层提取３２中特征
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 第二层卷积层提取６４种特征
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 再将第二个卷积层的输出进行变形再转为1D,然后再接上一个全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 防止过拟合设置dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 最后再接连一个softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()
for i in range(2000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# 输出：原例子里面循环20000次，由于时间过久，改成2000
# step 0, training accuracy 0.06
# step 100, training accuracy 0.78
# step 200, training accuracy 0.9
# step 300, training accuracy 0.84
# step 400, training accuracy 0.96
# step 500, training accuracy 0.96
# step 600, training accuracy 0.98
# step 700, training accuracy 0.98
# step 800, training accuracy 0.9
# step 900, training accuracy 0.94
# step 1000, training accuracy 0.92
# step 1100, training accuracy 0.98
# step 1200, training accuracy 0.96
# step 1300, training accuracy 0.98
# step 1400, training accuracy 0.98
# step 1500, training accuracy 1
# step 1600, training accuracy 0.98
# step 1700, training accuracy 0.98
# step 1800, training accuracy 0.94
# step 1900, training accuracy 0.98
# 2018-12-29 11:56:01.903823: W tensorflow/core/framework/allocator.cc:122] Allocation of 31360000 exceeds 10% of system memory.
# 2018-12-29 11:56:01.916497: W tensorflow/core/framework/allocator.cc:122] Allocation of 1003520000 exceeds 10% of system memory.
# 2018-12-29 11:56:11.699329: W tensorflow/core/framework/allocator.cc:122] Allocation of 250880000 exceeds 10% of system memory.
# 2018-12-29 11:56:13.103284: W tensorflow/core/framework/allocator.cc:122] Allocation of 501760000 exceeds 10% of system memory.
# 2018-12-29 11:56:23.461794: W tensorflow/core/framework/allocator.cc:122] Allocation of 125440000 exceeds 10% of system memory.
# test accuracy 0.9733


