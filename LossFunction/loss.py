# -*- coding:utf-8 -*-
#这个就是经典的损失函数，用于计算　交叉熵
#详细的交叉熵的使用可以看之前的demo
#cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0))+(1-y)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))

# 接下来介绍一下交叉熵里面的相关函数

#这个函数可以将一个张量中的数值限定在一个范围之内例如：1e-10, 1.0，小于1e-10的会变成1e-10,大于１的会变成1
#tf.clip_by_value(1-y, 1e-10, 1.0)

#这是个取对数的函数
#tf.log(tf.constant([1.0, 2.0, 3.0]))

#求平均值
#tf.reduce_mean()

#下面是损失函数的具体例子
import tensorflow as tf
import numpy as np
#每次喂入的数据
BATCH_SIZE = 8
#生成数据的随机种子
SEED = 23455
#生成32行2列的数据集
rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
#rdm.rand()会生成[0, 1)的随机数据(rdm.rand()/10.0-0.05)作为噪声范围-0.05~0.05
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]

#定义输入和输出
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

#定义使用均方误差的损失函数
loss_mse = tf.reduce_mean(tf.square(y_ - y))
#定义反向传播方法：梯度下降
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 20000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = (i*BATCH_SIZE) % 32 + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
		if i % 500 == 0:
			print("经过%d次训练，w1是："%(i))
			print(sess.run(w1))
	print("最后w1 is ",sess.run(w1))
#　省略...
# 经过18000次训练，w1是：
# [[0.9684917]
#  [1.0262802]]
# 经过18500次训练，w1是：
# [[0.9718707]
#  [1.0233142]]
# 经过19000次训练，w1是：
# [[0.974931 ]
#  [1.0206276]]
# 经过19500次训练，w1是：
# [[0.9777026]
#  [1.0181949]]
# 最后w1 is  [[0.98019385]
#  [1.0159807 ]]
#可见最后都是趋近于１的，符合Y_ = x1+x2+(rdm.rand()/10.0-0.05)这个函数