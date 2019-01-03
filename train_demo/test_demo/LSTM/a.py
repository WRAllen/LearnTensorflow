# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from matplotlib import style

 
origin_data = [1, 0, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 0, 1, 2, 1, 2, 1, 3, 3, 4, 2, 5, 2, 2, 5, 16, 2, 0, 0, 2, 4, 5, 3, 0, 1, 0, 3, 3, 0, 1, 3, 2, 1, 1, 2, 0, 2, 2, 4, 0, 5, 2, 4, 1, 2, 7, 10, 4, 3, 4, 1, 5, 4, 5, 11, 2, 5, 7, 1, 13, 1, 2, 3, 4, 6, 4, 5, 3, 6, 5, 8, 3, 1, 1, 7, 5, 3, 3, 3, 6, 3, 1, 2, 10, 4, 5, 2, 3, 1, 1, 11, 6, 2, 2, 7, 6, 5, 7, 12, 1, 3, 6, 13, 7, 5, 5, 1, 3, 7, 2, 9, 11, 4, 6, 19, 10, 6, 8, 9, 3, 0, 8, 11, 11, 10, 6, 6, 6, 7, 6, 10, 13, 35, 31, 25, 24, 10, 12, 5]

'''自定义数据尺度缩放函数'''
def data_processing(raw_data,scale=True):
    if scale == True:
        return (raw_data-np.mean(raw_data))/np.std(raw_data)#标准化
    else:
        return (raw_data-np.min(raw_data))/(np.max(raw_data)-np.min(raw_data))#极差规格化

'''设置绘图风格'''
style.use('ggplot')

'''设置隐层神经元个数'''
HIDDEN_SIZE = 40
'''设置隐层层数'''
NUM_LAYERS = 1
'''设置一个时间步中折叠的递归步数'''
TIMESTEPS = 5
'''设置训练轮数'''
TRAINING_STEPS = 2000
'''设置训练批尺寸'''
BATCH_SIZE = 20
'''样本数据生成函数'''
def generate_data(seq):
    X = []#初始化输入序列X
    Y= []#初始化输出序列Y
    '''生成连贯的时间序列类型样本集，每一个X内的一行对应指定步长的输入序列，Y内的每一行对应比X滞后一期的目标数值'''
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])#从输入序列第一期出发，等步长连续不间断采样
        Y.append([seq[i + TIMESTEPS]])#对应每个X序列的滞后一期序列值
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


'''定义LSTM cell组件，该组件将在训练过程中被不断更新参数'''
def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)#
    return lstm_cell

'''定义LSTM模型'''
def lstm_model(X, y):
    '''以前面定义的LSTM cell为基础定义多层堆叠的LSTM，我们这里只有1层'''
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])

    '''将已经堆叠起的LSTM单元转化成动态的可在训练过程中更新的LSTM单元'''
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    '''根据预定义的每层神经元个数来生成隐层每个单元'''
    output = tf.reshape(output, [-1, HIDDEN_SIZE])

    '''通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构'''
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    '''统一预测值与真实值的形状'''
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])

    '''定义损失函数，这里为正常的均方误差'''
    loss = tf.losses.mean_squared_error(predictions, labels)

    '''定义优化器各参数'''
    train_op = tf.contrib.layers.optimize_loss(loss,
                                               tf.contrib.framework.get_global_step(),
                                               optimizer='Adagrad',
                                               learning_rate=0.6)
    '''返回预测值、损失函数及优化器'''
    return predictions, loss, train_op

'''载入tf中仿sklearn训练方式的模块'''
learn = tf.contrib.learn

'''初始化我们的LSTM模型，并保存到工作目录下以方便进行增量学习'''
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir='Models/model_1'))

'''对原数据进行尺度缩放'''
data = data_processing(origin_data)

'''将所有样本来作为训练样本'''
train_X, train_y = generate_data(data)
print(train_X)
print("=====")
print(train_y)
print("######################")
'''将所有样本作为测试样本'''
test_X, test_y = generate_data(data)
print(test_X)
print("=====")
print(test_y)

'''以仿sklearn的形式训练模型，这里指定了训练批尺寸和训练轮数'''
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

'''利用已训练好的LSTM模型，来生成对应测试集的所有预测值'''
predicted = np.array([pred for pred in regressor.predict(test_X)])

'''绘制反标准化之前的真实值与预测值对比图'''
plt.plot(predicted, label='test')
plt.plot(test_y, label='realy')
plt.title('before')
plt.legend()
plt.show()


# '''自定义反标准化函数'''
# def scale_inv(raw_data,scale=True):

#     if scale == True:
#         return raw_data*np.std(origin_data )+np.mean(origin_data )
#     else:
#         return raw_data*(np.max(origin_data )-np.min(origin_data ))+np.min(origin_data )


# '''绘制反标准化之前的真实值与预测值对比图'''
# plt.figure()
# plt.plot(scale_inv(predicted), label='test')
# plt.plot(scale_inv(test_y), label='realy')
# plt.title('after')
# plt.legend()
# plt.show()