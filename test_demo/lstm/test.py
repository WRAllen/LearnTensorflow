# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from matplotlib import style

'''设置绘图风格'''
style.use('ggplot')


train_qty = [1, 1, 2, 2, 1, 0, 1, 2, 1, 2, 1, 3, 3, 4, 2, 5, 2, 2, 5, 16, 2, 0, 0, 2, 4, 5, 3, 0, 1, 0, 3, 3, 0, 1, 3, 2, 1, 1, 2, 0, 2, 2, 4, 0, 5, 2, 4, 1, 2, 7, 10, 4, 3, 4, 1, 5, 4, 5, 11, 2, 5, 7, 1, 13, 1, 2, 3, 4, 6, 4, 5, 3, 6, 5, 8, 3, 1, 1, 7, 5, 3, 3, 3, 6, 3, 1, 2, 10, 4, 5, 2, 3, 1, 1, 11, 6, 2, 2, 7, 6, 5, 7, 12, 1, 3, 6, 13, 7, 5, 5, 1, 3, 7, 2, 9, 11, 4, 6, 19, 10, 6, 8, 9, 3, 0, 8, 11, 11, 10, 6, 6, 6, 7, 6, 10, 13, 35, 31, 25, 24, 10, 12, 5, 16, 12, 4, 8, 5, 12, 7, 18, 4, 6, 5, 4, 16, 11, 20, 7, 7, 4, 1, 3, 4, 6, 1, 4, 5, 1, 11, 3, 2, 1, 3, 3, 1, 3, 3, 10, 3, 4, 2, 1, 6, 4, 1, 2, 2, 1, 1, 7, 4, 4, 1, 1, 3, 0, 8, 1, 3]

test_qty = [4, 0, 2, 8, 3, 1, 3, 1, 1, 5, 1, 2, 1, 1, 3, 0, 3, 3, 2, 3, 2, 1, 1, 3, 0, 2, 4, 1, 0, 0, 1, 3, 0, 1, 2, 1, 1, 2, 4, 1, 0, 3, 0, 0, 2, 1, 0, 2, 1, 1]

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
        start=0
        end=start+batch_size
        while(end<len(train_x)):
            _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
            start+=batch_size
            end=start+batch_size 
    saver.save(sess,'my_net/net.ckpt')
#————————————————预测模型————————————————————
def prediction():
    pred,_=lstm(1)      #预测时只输入[1,time_step,input_size]的测试数据
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

print("开始训练")
# train_lstm()
# print("end")
print("开始预测")
prediction() 
print("end")
