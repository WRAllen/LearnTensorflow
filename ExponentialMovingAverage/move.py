# -*- coding:utf-8 -*-
import tensorflow as tf

#定义一个变量用于计算滑动平均，这个变量的初始值设置为0，并且滑动平均的变量必须是实数型
w1 = tf.Variable(0, dtype=tf.float32)
#迭代轮数，不可训练trainable设置为false
global_step = tf.Variable(0, trainable=False)
#定义滑动平均类，给滑动衰减率
MOVING_AVERAGE_DECAY = 0.99
#定义一个滑动平均的类．初始化的时候要给定衰减率(0.99,一般衰减率都要非常的接近1．)，和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#自动将所有待训练的参数汇总到列表ema_op = ema.apply([w1])
ema_op = ema.apply(tf.trainable_variables())
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print("初始化")
	print(sess.run([w1, ema.average(w1)]))
	print("更新变量w1的值到5")
	sess.run(tf.assign(w1, 5))
	#更新了变量的值也会导致影子变量和衰减率的变化
	#衰减率min{0.99, (1 + step)/(10 + step)}因为这里step = 0,所以衰减率decay = 0.1
	#相应的滑动平均(影子变量)变为 0.1*0 + 0.9*5 = 4.5
	sess.run(ema_op)
	print(sess.run([w1, ema.average(w1)]))
	print("更新step和w1的值")
	sess.run(tf.assign(global_step, 10000))
	#更新w1的值
	sess.run(tf.assign(w1, 10))
	#更新了变量的值也会导致影子变量和衰减率的变化
	#衰减率min{0.99, (1 + step)/(10 + step)}因为这里step = 10000,所以衰减率decay = 0.99
	#相应的滑动平均(影子变量)变为 0.99*4.5 + 0.01*10 = 4.555
	sess.run(ema_op)
	print(sess.run([w1, ema.average(w1)]))
	print("再次更新滑动平均")
	sess.run(ema_op)
	print(sess.run([w1, ema.average(w1)]))
# 输出
# 初始化
# [0.0, 0.0]
# 更新变量w1的值到5
# [5.0, 4.5]
# 更新step和w1的值
# [10.0, 4.555]
# 再次更新滑动平均
# [10.0, 4.60945]
