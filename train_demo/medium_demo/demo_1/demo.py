# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("path/to/MNIST_data/", one_hot=True)
#定义输入(28*28=784)和输出层(一个一维长度为10的数组，里面只有0,1代表是数字几)
INPUT_NODE = 784
OUTPUT_NODE = 10
#配置神经网络的参数
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8 #基本学习速率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率
REGULARIZATION_RATE = 0.0001 #正则化项在损失函数里面的系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
	#判断是否提供滑动平均类
	if avg_class:
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
		return tf.matmul(layer1, weight2) + biases2
	else:
		layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class(weight1)) + avg_class(biases1))
		return tf.matmul(layer1, avg_class(weight2)) + avg_class(biases2)

def train(mnist):
	x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
	y_ = tf.placehoder(tf.float32, [None, OUTPUT_NODE], name="y-input")
	#隐藏层
	weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
	biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
	#输出层
	weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
	biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
	#计算当前参数下神经网络前向传播的结果，这里不使用滑动平均值
	y = inference(x, None, weight1, biases1, weight2, biases2)
	#定义训练轮次
	global_step = tf.Variable(0, trainable=False)
	#初始化滑动平均类
	variable_averages = tf.train.ExponetialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	average_y = inference(x, variable_averages, weight1, biases1, weight2, biases2)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce.mean(cross_entropy)
	#计算L2正则损失函数
	regularizer = rf.contrib.layers.l2.regularizer(REGULARIZATION_RATE)
	regularization = regularizer(weight1)+regularizer(weight2)
	loss = cross_entropy_mean + regularization
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	train_op = tf.group(train_step, variable_averages_op)
	correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#初始化会话和开始训练
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
		test_feed = {x:mnist.test.images,y_:mnist.test.labels}
		for i in range(TRAINING_STEPS):
			if i % 1000 == 0:
				validate_acc = sess.run(accuracy, feed_dict=validate_feed)
				print("经过%d次训练，使用滑动平均%g"%(i,validate_acc))
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op, feed_dict={x:xs, y_:ys})
		test_acc = sess.run(accuracy, feed_dict=test_feed)
		print("经过%d次训练，测试使用滑动平均%g"%(TRAINING_STEPS,test_acc))

def main(argv=None):
	mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()
