#conding:utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from matplotlib import style
import matplotlib.pyplot as plt
#超参数
INPUT_NODE = 2
OUTPUT_NODE = 1
LAYER_NODE = 10
BATCH_SIZE = 10
train_row = 90
test_row = 10

MODEL_SAVE_PATH = './my_net'
MODEL_NAME = 'test_model'
#定义计数器
global_step = tf.Variable(0, trainable=False)


def pre_full(datas):
    X = tf.placeholder(tf.float32, [None, INPUT_NODE])
    Y = tf.placeholder(tf.float32, [None, OUTPUT_NODE])

    #隐藏层
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[1, LAYER_NODE]))
    #输出层
    weight2 = tf.Variable(tf.truncated_normal([LAYER_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[1, OUTPUT_NODE]))

    layer_int = tf.nn.sigmoid(tf.matmul(X, weight1) + biases1)
    layer_out = tf.matmul(layer_int, weight2) + biases2

    saver = tf.train.Saver()

    correct_prediction = tf.equal(layer_out, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #增加断点续训的功能
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)


        # accuracy_score = sess.run(accuracy, feed_dict={X:datas.values[90:100,2:].tolist(), Y:datas.values[90:100,1].reshape([10,1])})
        accuracy_score = sess.run(accuracy, feed_dict={X:datas[:90,1:].tolist(), Y:datas[:90,0].reshape([90,1])})
        print("测试数据的准确度:%f"%(accuracy_score))

        '''设置绘图风格'''
        style.use('ggplot')
        #以折线图表示结果
        plt.figure()
        # 正常的训练数据
        base_x_lable = list(range(len(datas[:90,0])))
        total_x_lable = list(range(len(datas[:90,0]), len(datas[:90,0])+len(datas[90:100,0])))
        plt.plot(base_x_lable, datas[:90,0], label='train', color='r')
        #真实值
        # plt.plot(total_x_lable, datas.values[80:100,1], label='realy', color='b')
        #测试训练数据
        plt.plot(base_x_lable, sess.run(layer_out, feed_dict={X:datas[:90,1:].tolist(), Y:datas[:90,0].reshape([90,1])}), label='pre', color='g')
        #预测值
        # plt.plot(total_x_lable, sess.run(layer_out, feed_dict={X:datas.values[80:100,2:].tolist(), Y:datas.values[80:100,1].reshape([20,1])}), label='pre', color='g')
        plt.show()

#标准化数据
def format_data(datas, standard=None):
    datas = datas.iloc[:,1:4].values   #取第2-4列数据
    return (datas-np.mean(datas))/np.std(datas) if standard else datas

datas = format_data(pd.read_csv("data.csv"), True)
print("开始预测 ")
pre_full(datas)
print("预测结束")

