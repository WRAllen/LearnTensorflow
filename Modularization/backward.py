# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateds
import forward

STEPS = 40000
BATCH_SIZE = 30
#指数衰减学习速超参数
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
LEARNING_RATE_STEP = 300/BATCH_SIZE
#定义正则化权重
REGULARIZER = 0.01

def backward():
	x = tf.placeholder(tf.float32, shape=(None, 2))
	y_ = tf.placeholder(tf.float32, shape=(None, 1))

	X, Y_, Y_c = generateds.generateds()
	
	y = forward.forward(x, REGULARIZER)

	global_step = tf.Variable(0, trainable=False)

	#定义指数衰减学习率
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, 
		LEARNING_RATE_DECAY, staircase=True)

	#定义损失函数
	loss_mse = tf.reduce_mean(tf.square(y-y_))
	loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

	#定义方向传播方法:含正则化
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

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

		plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
		plt.contour(xx, yy, probs, levels=[.5])
		plt.show()

if __name__=='__main__':
	backward()
