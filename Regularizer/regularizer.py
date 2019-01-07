# -*- coding:utf-8 -*-
#下面是正则化的具体例子
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 30
seed = 2
#生成基于seed的随机函数
rdm = np.random.RandomState(seed)
#随机返回300行2列的矩阵，代表300组坐标点(x0,x1)作为输入数据集
X = rdm.randn(300, 2)
#从这组矩阵中取出满足函数条件的数据赋予1,否则赋予0
Y_ = [int(x0*x0 + x1*x1 < 2) for (x0, x1) in X ]
#1赋予red，0赋予blue
Y_c = [['red' if y else 'blue'] for y in Y_ ]
#对数据进行变形,-1表示n行
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)
print(X)
print(Y_)
print(Y_c)
#X[:,0]表示取X的第一列
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.show()
#定义神经网络的输入和输出还有参数，定义前向传播过程
def get_weight(shape, regularizer):
	w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.constant(0.01, shape=shape))
	return b

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
#隐藏层
w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
#这边使用了relu激活函数
y1 = tf.nn.relu(tf.matmul(x, w1)+b1)
#输出层
w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2)+b2#输出层不需要激活函数
#定义损失函数
loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

#定义方向传播方法:不含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 40000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 300
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
		if i % 2000 == 0:
			loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
			print("经过%d次训练，loss是：%f"%(i, loss_mse_v))
	#xx,yy在-3到3之间步长0.01，生成二维网格坐标点
	xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
	#将xx,yy拉直，合并成一个2列的矩阵
	grid = np.c_[xx.ravel(), yy.ravel()]
	probs = sess.run(y, feed_dict={x:grid})
	#probs的shape调整成xx的样子
	probs = probs.reshape(xx.shape)
	print("w1",sess.run(w1))
	print("b1",sess.run(b1))
	print("w2",sess.run(w2))
	print("b2",sess.run(b2))

plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()




#定义方向传播方法:含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 40000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 300
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
		if i % 2000 == 0:
			loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
			print("经过%d次训练，loss是：%f"%(i, loss_mse_v))
	#xx,yy在-3到3之间步长0.01，生成二维网格坐标点
	xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
	#将xx,yy拉直，合并成一个2列的矩阵
	grid = np.c_[xx.ravel(), yy.ravel()]
	probs = sess.run(y, feed_dict={x:grid})
	#probs的shape调整成xx的样子
	probs = probs.reshape(xx.shape)
	print("w1",sess.run(w1))
	print("b1",sess.run(b1))
	print("w2",sess.run(w2))
	print("b2",sess.run(b2))

plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
