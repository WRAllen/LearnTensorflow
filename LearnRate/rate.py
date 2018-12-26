# -*- coding:utf-8 -*-
import tensorflow as tf 
global_step = tf.Variable(10)
#通过exponential_decay函数生成学习速率
learning_rate = tf.train.exponential_decay(0.1, global_step, 10, 0.96, staircase=True)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	print(sess.run(learning_rate))
#输出
# 0.096

