# -*- coding:utf-8 -*-
import tensorflow as tf
#这是一个获得一层神经网络边上权重，并且把这个权重的L2正则化损失加入名称为"losses"的集合中去
def get_weight(shape, lambda_weight):
	var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
	#add_to_collection函数将这个新生成的变量的L2正则化损失加入集合
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_weight)(var))
	return var

batch_size = 8
#两个输入
x = tf.placeholder(tf.float32, shape=(None, 2), name="X_input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="Y_input")
#定义每一层网络中的节点个数
layer_dimension = [2, 10, 10, 10, 1]
#神经网络的层数
n_layers = len(layer_dimension)

#当前的层
cur_layer = x
#当前层的节点(现在是初始层节点数)
in_dimension = layer_dimension[0]

#生成５层的全链接神经网络(只循环4次，因为初始层已经初始化好了)
for i in range(1, n_layers):
	out_dimension = layer_dimension[i]
	weight = get_weight([in_dimension, out_dimension], 0.001)
	bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
	#使用ReLU激活函数（具体为什么要使用母鸡）
	cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight)+bias)
	in_dimension = layer_dimension[i]
mse_loss = tf.reduce_mean(tf.square(y_-cur_layer))
tf.add_to_collection('losses', mse_loss)
loss = tf.add_n(tf.get_collection('losses'))

