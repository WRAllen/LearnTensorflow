#conding:utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import Normalizer
#超参数
INPUT_NODE = 2
OUTPUT_NODE = 1
LAYER_NODE = 10
STEPS = 1500
BATCH_SIZE = 5
train_row = 90
test_row = 10
#最初学习率
LEARNING_RATE_BASE = 0.01
#学习衰减率
LEARNING_RATE_DECAY = 0.99
#喂入多少轮BATCH_SIZE后更新一次学习率，一般为:样本总数/BATCH_SIZE
LEARNING_RATE_STEP = 1
#定义计数器
global_step = tf.Variable(0, trainable=False)
#定义指数衰减学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, 
        LEARNING_RATE_DECAY, staircase=True)
#未优化学习速率前的值
# learning_rate = 0.001
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99
#定义正则化权重
REGULARIZER = 0.0001

MODEL_SAVE_PATH = './my_net'
MODEL_NAME = 'test_model'



def train_full(datas):
    X = tf.placeholder(tf.float32, [None, INPUT_NODE], name="X")
    Y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="Y")

    #隐藏层
    weight1 = tf.Variable(tf.random_normal([INPUT_NODE, LAYER_NODE], stddev=0.1))
    # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(weight1))
    biases1 = tf.Variable(tf.zeros(shape=[LAYER_NODE]) + 0.1)
    #输出层
    weight2 = tf.Variable(tf.random_normal([LAYER_NODE, OUTPUT_NODE], stddev=0.1))
    # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(weight2))
    biases2 = tf.Variable(tf.zeros(shape=[OUTPUT_NODE]) + 0.1)

    ## 丢弃样本比例，1是所有的样本使用，这个好像是自动回丢弃不靠谱的样本
    keep_prob_s = tf.placeholder(dtype=tf.float32, name="keep_prob_s")

    layer_int = tf.nn.relu(tf.matmul(X, weight1) + biases1)
    #隐藏层到输出层丢弃不靠谱数据
    layer_int = tf.nn.dropout(layer_int, keep_prob=keep_prob_s)  
    layer_out = tf.add(tf.matmul(layer_int, weight2),biases2, name="layer_out")
    #正则化损失函数
    # loss = tf.reduce_mean(tf.square(layer_out-Y)) + tf.add_n(tf.get_collection('losses'))
    loss = tf.reduce_mean(tf.square(layer_out-Y))
    #反方向传播函数，这里不加global_step=global_step，在保存时就不会有断点续训
    #第一次使用的反向传播函数
    # train_step = tf.train.GradientDescentOptimizerr(learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #定义滑动平
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
            for step in range(train_row-BATCH_SIZE):
               _, loss_value = sess.run([train_op, loss], feed_dict={X:datas[step:step+BATCH_SIZE,1:].tolist(), Y:datas[step:step+BATCH_SIZE,0].reshape([BATCH_SIZE,1]), keep_prob_s: 1}) 
            if i % 10 == 0:
                print("loss是：%f"%(loss_value))
            if i % 500 == 0:
                print("经过%d轮训练，loss是：%f"%(i, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

#标准化数据
def format_data(datas, standard=None):
    datas = datas.iloc[:,1:4].values   #取第2-4列数据
    return (datas-np.mean(datas))/np.std(datas) if standard else datas


datas = format_data(pd.read_csv("data.csv"), True)
# print(datas)
print("开始训练")
train_full(datas)
print("训练结束")

