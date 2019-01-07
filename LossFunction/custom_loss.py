# -*- coding:utf-8 -*-
#之前的cross_entropy其实就是一个损失函数，主要用于度量两个概率分布间的差异性信息
#我们也可以根据具体情况自定义相关的损失函数
#基本用了ForwardPropagation&BackPropagation里面的demo

#背景：是预测商品的销量，加上预测多了则损失商品的成本（１元），预测少了损失盈利（１０元）
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
SEED = 1
#两个输入
x = tf.placeholder(tf.float32, shape=(None, 2), name="X_input")

y_ = tf.placeholder(tf.float32, shape=(None, 1), name="Y_input")

#只做简单的加权和
w1 = tf.Variable(tf.random_normal((2, 1), stddev=1, seed=1))
y = tf.matmul(x, w1)

#定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1

#定义损失函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*loss_more, (y_-y)*loss_less))
#tf.where(x,y1,y2):如果x为true则运行y1,如果x为false则运行y2
#tf.greater(x, y):输入的x,y是两个张量，会比较这两个张量每一个元素的大小，返回比较结果

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
#生成一个模拟数据集
rdm = RandomState(SEED)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 +rdm.rand()/10.0 -0.05] for (x1, x2) in X]

#开始训练
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	STEP = 5000
	for i in range(STEP):
		start = (i*batch_size)%dataset_size
		end = min(start+batch_size, dataset_size)
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		print(sess.run(w1))

# [[1.0194283]
#  [1.0428752]]
# [[1.0194151]
#  [1.0428821]]
# [[1.019347 ]
#  [1.0428089]]
#可见两个参数都大于1了，模型默认往多了预测，交换loss_less,loss_more的值，模型就会往少了预测
# [[0.9548402]
#  [0.9813122]]
# [[0.95491844]
#  [0.9814671 ]]
# [[0.95506585]
#  [0.98148215]]
# [[0.9552581]
#  [0.9813394]]
