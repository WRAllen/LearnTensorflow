# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from demo_data import train_qty
import os
#标准化数据
normalize_data = (train_qty-np.mean(train_qty))/np.std(train_qty)
#增加维度,[x,x,x]变成[[x], [x], [x]]
normalize_data = normalize_data[:,np.newaxis]
sess = tf.InteractiveSession()
#生成训练集
#设置常量
time_step=10      #时间步
rnn_unit=10       #hidden layer units
batch_size=20     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.0006         #学习率
STEPS = 20000
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99
#定义正则化权重
REGULARIZER = 0.0001
MODEL_SAVE_PATH = './my_net'
MODEL_NAME = 'mnist_model'

train_x,train_y=[],[]   #训练集
#为什么y要比x多一个位置，因为y是通过x来预测的
for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist()) 

#——————————————————定义神经网络变量——————————————————
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   #每批次tensor对应的标签
#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }

#——————————————————定义神经网络变量——————————————————
def lstm(batch):      
    #参数：输入网络批次数目
    w_in=weights['in']
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(w_in))
    b_in=biases['in']
    #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input=tf.reshape(X,[-1,input_size])
    input_rnn=tf.matmul(input,w_in)+b_in
    #将tensor转成3维，作为lstm cell的输入
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit]) 
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    #作为输出层的输入
    output=tf.reshape(output_rnn,[-1,rnn_unit])
    w_out=weights['out']
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(w_out))
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred, final_states

#——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    global_step = tf.Variable(0, trainable=False)
    pred, _=lstm(batch_size)
    #损失函数(包涵正则化)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1]))) + tf.add_n(tf.get_collection('losses'))

    #反方向传播函数，这里不加global_step=global_step，在保存时就不会有断点续训
    # train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
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
            start = 0
            end = start + batch_size
            while(end <= len(train_x)):
                _, loss_value, step=sess.run([train_op, loss, global_step], feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start += batch_size
                end = start + batch_size 
            if i % 1000 == 0:
                print("经过%d轮训练，loss是：%f"%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

print("开始训练")
train_lstm()
print("end")

