#conding:utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import Normalizer

train_data = np.loadtxt(open("data.csv", "rb"), delimiter=",", skiprows=1)
## 获取除第3列数据，第3列为结果值（销量），写列号（第几列作为特征值）为笨方法，不知道该怎么写
#以第二列的时间作为特征
tezheng = train_data[:89,[1, 2]]    
tf.reset_default_graph()
## 销量结果，用于训练
sale = train_data[:89, 0]
## 将特征进行归一化，使训练的时候不会受大小特征影响严重，注：非scale，scale是对于列的，scale再进行预测的时候即使相同一行数据，因为其它数据变了，这一列有变，scale之后得到的值也不一样，预测结果自然也不一样了。
## 下面这种是对行的归一化
X_train = Normalizer().fit_transform(tezheng)   ## 多维特征
y_train = sale.reshape((-1,1))        ## 结果，从一维转为二维

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


X = tf.placeholder(tf.float32, [None, 2], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

#隐藏层
weight1 = tf.Variable(tf.random_normal([2, 10]))
# tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(weight1))
biases1 = tf.Variable(tf.zeros(shape=[1, 10]) + 0.1)
#原来biases1 = tf.Variable(tf.zeros(shape=[LAYER_NODE]) + 0.1)
#输出层
weight2 = tf.Variable(tf.random_normal([10, 1]))
# tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(weight2))
biases2 = tf.Variable(tf.zeros(shape=[1, 1]) + 0.1)

## 丢弃样本比例，1是所有的样本使用，这个好像是自动回丢弃不靠谱的样本
keep_prob_s = tf.placeholder(dtype=tf.float32, name="keep_prob_s")

layer_int = tf.nn.sigmoid(tf.matmul(X, weight1) + biases1)
# 原来layer_int = tf.nn.relu(tf.matmul(X, weight1) + biases1)

#隐藏层到输出层丢弃不靠谱数据
layer_int = tf.nn.dropout(layer_int, keep_prob=keep_prob_s)  
layer_out = tf.matmul(layer_int, weight2) + biases2
#正则化损失函数
# loss = tf.reduce_mean(tf.square(layer_out-Y)) + tf.add_n(tf.get_collection('losses'))
# 原来loss = tf.reduce_mean(tf.square(layer_out-Y))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y - layer_out), reduction_indices=[1]))
#反方向传播函数，这里不加global_step=global_step，在保存时就不会有断点续训
#第一次使用的反向传播函数
# train_step = tf.train.GradientDescentOptimizerr(learning_rate).minimize(loss, global_step=global_step)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
# 原来train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
# #定义滑动平
# ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
# #把所有可训练的变量生成该变量对应的滑动平均
# ema_op = ema.apply(tf.trainable_variables())
# #指定依赖关系,可以理解为train_step的变量依赖于滑动平均
# with tf.control_dependencies([train_step, ema_op]):
#     train_op = tf.no_op(name='train')

saver = tf.train.Saver()


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #增加断点续训的功能
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    feed_dict_train = {inputX: X_train, y_true: y_train, keep_prob_s: 1}
    for i in range(STEPS):
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict_train) 
        if i % 1000 == 0:
            print("经过%d轮训练，loss是：%f"%(i, loss_value))
    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)



