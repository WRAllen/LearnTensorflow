# -*- coding:utf-8 -*-
#之前的cross_entropy其实就是一个损失函数，主要用于度量两个概率分布间的差异性信息
#我们也可以根据具体情况自定义相关的损失函数
#基本用了ForwardPropagation&BackPropagation里面的demo

#背景：是预测商品的销量，加上预测多了则损失商品的成本（１元），预测少了损失盈利（１０元）
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
#两个输入
x = tf.placeholder(tf.float32, shape=(None, 2), name="X_input")
# 一个输出(为预测值)
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="Y_input")

#只做简单的加权和
w1 = tf.Variable(tf.random_normal((2, 1), stddev=1, seed=1))
y = tf.matmul(x, w1)

#定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1

#定义损失函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*loss_more, (y_-y)*loss_less)
#tf.where(x,y1,y2):如果x为true则运行y1,如果x为false则运行y2
#tf.greater(x, y):输入的x,y是两个张量，会比较这两个张量每一个元素的大小，返回比较结果


