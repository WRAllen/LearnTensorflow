# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from demo_data import train_qty
#标准化数据
normalize_data = (train_qty-np.mean(train_qty))/np.std(train_qty)
#增加维度,变成[[xxx]]
normalize_data = normalize_data[:,np.newaxis]
sess = tf.InteractiveSession()
#生成训练集
#设置常量
time_step=20      #时间步
rnn_unit=10       #hidden layer units
batch_size=60     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.0006         #学习率
train_x,train_y=[],[]   #训练集
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
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    pred,_=lstm(batch_size)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    #重复训练2000次
    for i in range(2000):
        if i%500 == 0:
            print("第",i,"次训练")
        start=0
        end=start+batch_size
        while(end<len(train_x)):
            _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
            start+=batch_size
            end=start+batch_size 
    saver.save(sess,'my_net/net.ckpt')

print("开始训练")
train_lstm()
print("end")

