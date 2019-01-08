#conding:utf-8
import tensorflow as tf
import pandas as pd
import os
from matplotlib import style
import matplotlib.pyplot as plt
#超参数
INPUT_NODE = 2
OUTPUT_NODE = 1
LAYER_NODE = 10
learning_rate = 0.001
STEPS = 20000
BATCH_SIZE = 10
train_row = 90
test_row = 10

MODEL_SAVE_PATH = './my_net'
MODEL_NAME = 'test_model'



def pre_full(datas):
    X = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    Y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    #隐藏层
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER_NODE]))
    #输出层
    weight2 = tf.Variable(tf.truncated_normal([LAYER_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    layer_int = tf.nn.relu(tf.matmul(X, weight1) + biases1)
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


        accuracy_score = sess.run(accuracy, feed_dict={X:datas.values[90:100,2:].tolist(), Y:datas.values[90:100,1].reshape([10,1])})
        print("测试数据的准确度:%f"%(accuracy_score))

        '''设置绘图风格'''
        style.use('ggplot')
        #以折线图表示结果
        plt.figure()
        # 正常的训练数据
        print(list(range(len(datas.values[:90,1]))))
        plt.plot(list(range(len(datas.values[:90,1]))), datas.values[:90,1], label='train', color='r')
        plt.plot([90, 91, 92, 93, 94, 95, 96, 97, 98, 99], datas.values[90:100,1], label='realy', color='b')
        plt.plot([90, 91, 92, 93, 94, 95, 96, 97, 98, 99], sess.run(layer_out, feed_dict={X:datas.values[90:100,2:].tolist(), Y:datas.values[90:100,1].reshape([10,1])}), label='pre', color='g')
        plt.show()




datas = pd.read_csv("data.csv")
print("开始预测 ")
pre_full(datas)
print("预测结束")

