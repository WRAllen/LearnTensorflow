# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
#定义正则化权重
REGULARIZER = 0.0001
STEPS = 50000
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './my_net'
MODEL_NAME = 'mnist_model'

def backward(mnist):
	x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
	y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
	y = forward.forward(x, REGULARIZER)
	global_step = tf.Variable(0, trainable=False)
	#定义交叉熵,和使用了softmax函数,并且使用了正则化
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	cem = tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection('losses'))
	#定义指数衰减学习率
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 
			mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)
	#使用梯度下降的方向传播函数
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	#定义滑动平均
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	#把所有可训练的变量生成该变量对应的滑动平均
	ema_op = ema.apply(tf.trainable_variables())
	#指定依赖关系,可以理解为train_step的变量依赖于滑动平均
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name='train')

	saver = tf.train.Saver()

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		#增加断点续训的功能
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		for i in range(STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})
			if i % 1000 == 0:
				print("经过%d次训练，loss是：%f"%(step, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
	mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
	backward(mnist)

if __name__ == '__main__':
	main()

