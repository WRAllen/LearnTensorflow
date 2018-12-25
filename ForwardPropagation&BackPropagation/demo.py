# -*- coding:utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState
#定义训练集的大小(就是一次传入几个样例数据)
batch_size = 8

###seed设置成一样保证每次生产的随机数都是一样的
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

#x可看成是用户的输入,这里和batch_size的大小有关，设置成None方便不同的batch_size的大小
#y_可与看做是预计输出结果
x = tf.placeholder(tf.float32, shape=(None, 2), name="X_input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="Y_input")

#定义神经网络的前向传播过程
#具体是x->a->y:a是隐藏层，这里是只有3层的神经网络（输入层－隐藏层－输出层）
#y是实际运算的结果
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


#通过sigmoid函数将y转化成0~1之内的数值，y代表正样本的数据，1-y代表负样本
y = tf.sigmoid(y)
#定义损失函数
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0))+(1-y)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
#定义学习速率
learning_rate = 0.001
#定义向传播算法
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#生成模拟输入数据,X,Y为输入数据和输出数据
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	print(sess.run(w1))
	print("=====")
	print(sess.run(w2))
	STEPS = 5000
	for i in range(STEPS):
		#每次选取batch_size个样本进行测试并且在dataset_size的范围之内
		#start=0,8,16...120 end=8,16...128
		start = (i*batch_size)%dataset_size
		end = min(start+batch_size, dataset_size)
		#通过选取的样本训练神经网络并更新参数
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i%1000 == 0:
			total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X,y_:Y})
			print("经过%d次的训练,交叉熵是:%g"%(i,total_cross_entropy))
	print("经过训练之后的结果")
	print(sess.run(w1))
	print("=====")
	print(sess.run(w2))

#输出
'''
[[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
=====
[[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]
经过0次的训练,交叉熵是:0.314006
经过1000次的训练,交叉熵是:0.0684551
经过2000次的训练,交叉熵是:0.033715
经过3000次的训练,交叉熵是:0.020558
经过4000次的训练,交叉熵是:0.0136867
经过训练之后的结果
[[-2.548655   3.0793087  2.8951712]
 [-4.1112747  1.6259071  3.3972702]]
=====
[[-2.3230937]
 [ 3.3011687]
 [ 2.4632082]]
'''