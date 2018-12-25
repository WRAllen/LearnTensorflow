# -*- coding:utf-8 -*-
import tensorflow as tf
###seed设置成一样保证每次生产的随机数都是一样的
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))
#x可看成是用户的输入
x = tf.constant([[0.7, 0.9]])
#具体是x->a->y:a是隐藏层，这里是只有3层的神经网络（输入层－隐藏层－输出层）
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y))

#输出[[3.957578]]
#或者ｘ的值可是先设置成待订具体看forward_input.py
