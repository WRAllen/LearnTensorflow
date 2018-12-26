# -*- coding:utf-8 -*-
#仅供查看，不可运行
import tensorflow as tf
#运行时需要把下面这三句注释掉
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, 2)
loss = tf.reduce_mean(tf.square(y_-y))+tf.contrib.layers.l2_regularizer(lambda_weight)(w)

#上面那个就符合j(x)+yR(w)
#之前的损失函数是不带+后面的yR(w)也就是tf.contrib.layers.l2_regularizer(lambda_weight)(w)
#y其实就是lambda_weight(正则化项的权重),R(w)就是w(也就是需要计算正则损失的参数)

#下面是具体的例子
weights = tf.constant([[1.0, 2.0], [3.0, 4.0]])
with tf.Session() as sess:
	#.5 = 0.5
	#计算方式：各绝对值相加x0.5
	print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
	#计算方式：各平方相加/2再x0.5(/2是为了求导得到的结果更加整洁?-不理解)
	print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))
#输出
# 5.0
# 7.5

