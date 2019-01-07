# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from matplotlib import style
from demo_data import train_qty, test_qty

#标准化数据
normalize_data = (train_qty-np.mean(train_qty))/np.std(train_qty)
#增加维度,变成[[xxx]]
normalize_data = normalize_data[:,np.newaxis]
sess = tf.InteractiveSession()

normalize_test = (test_qty-np.mean(test_qty))/np.std(test_qty)
normalize_test = normalize_test[:,np.newaxis]

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
def lstm(batch):      #参数：输入网络批次数目
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#————————————————预测模型————————————————————
def prediction():
    #预测时只输入[1,time_step,input_size]的测试数据
    pred,_=lstm(1)      
    saver = tf.train.Saver()
    saver.restore(sess, "my_net/net.ckpt")
    #取训练集最后一行为测试样本。shape=[1,time_step,input_size]
    prev_seq=train_x[-1]
    predict=[]
    #得到之后35个预测结果
    for i in range(50):
        next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
        predict.append(next_seq[-1])
        #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
        prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
    '''设置绘图风格'''
    style.use('ggplot')
    #以折线图表示结果
    plt.figure()
    # 正常的训练数据
    plt.plot(list(range(len(normalize_data))), normalize_data, label='train', color='r')
    # 测试未来35天的销量
    plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, label='test', color='b')
    # 实际未来35天的销量
    plt.plot(list(range(len(normalize_data), len(normalize_data) + len(normalize_test))), normalize_test, label='realy', color='r')
    plt.title('Test')
    plt.show()

print("开始预测")
prediction() 
print("end")
