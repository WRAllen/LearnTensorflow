from __future__ import print_function
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import numpy as np


## 从文件获取训练数据,其中skiprows为忽略的函数，即文件无关数据的行数，delimiter：str, default None定界符，备选分隔符（如果指定该参数，则sep参数失效）
train_data = np.loadtxt(open("data.csv", "rb"), delimiter=",", skiprows=1)

## 获取除第3列数据，第3列为结果值（销量），写列号（第几列作为特征值）为笨方法，不知道该怎么写
#以第二列的时间作为特征
tezheng = train_data[:89,[0, 2]]    

tf.reset_default_graph()

## 销量结果，用于训练
sale = train_data[:89, 1]
## 将特征进行归一化，使训练的时候不会受大小特征影响严重，注：非scale，scale是对于列的，scale再进行预测的时候即使相同一行数据，因为其它数据变了，这一列有变，scale之后得到的值也不一样，预测结果自然也不一样了。
## 下面这种是对行的归一化
X_train = Normalizer().fit_transform(tezheng)   ## 多维特征
y_train = sale.reshape((-1,1))        ## 结果，从一维转为二维

print(X_train, y_train)

#定义正则化权重
REGULARIZER = 0.0001

### 开始进行图的构建

# 特征与结果的替代符，声明类型，维度 ，name是用来生成模型之后，使用模型的时候调用用的
inputX = tf.placeholder(shape=[None, X_train.shape[1]], dtype=tf.float32, name="inputX")
y_true = tf.placeholder(shape=[None,1], dtype=tf.float32, name="y_true")

## 丢弃样本比例，1是所有的样本使用，这个好像是自动回丢弃不靠谱的样本
keep_prob_s = tf.placeholder(dtype=tf.float32, name="keep_prob")

### 第一层，一个隐藏层 开始
## shape的第一维就是特征的数量，第二维是给下一层的输出个数,  底下的矩阵相乘实现的该转换
Weights1 = tf.Variable(tf.random_normal(shape=[2, 10]), name="weights1")  ## 权重
#tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(Weights1)) #正则化
biases1 = tf.Variable(tf.zeros(shape=[1, 10]) + 0.1, name="biases1")       ## 偏置
### 第一层结束

### 第二层开始，即输出层
## 上一层的10，转为1，即输出销售量
Weights2 = tf.Variable(tf.random_normal(shape=[10, 1]), name="weights2")   ## 权重
#tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(Weights2)) #正则化
biases2 = tf.Variable(tf.zeros(shape=[1, 1]) + 0.1, name="biases2")        ## 偏置

## matmul矩阵相乘，nn.dropout 丢弃部分不靠谱数据
Wx_plus_b1 = tf.matmul(inputX, Weights1)
Wx_plus_b1 = tf.add(Wx_plus_b1, biases1)
Wx_plus_b1 = tf.nn.dropout(Wx_plus_b1, keep_prob=keep_prob_s)    

## 将结果曲线化，通常说非线性化
l1 = tf.nn.sigmoid(Wx_plus_b1, name="l1")

## matmul矩阵相乘 ,l1 为上一层的结果
Wx_plus_b2 = tf.matmul(l1, Weights2)
prediction = tf.add(Wx_plus_b2, biases2, name="pred")      ## pred用于之后使用model时进行恢复

learning_rate_base = 0.01 #最初学习率
learning_rate_decay = 0.99 #最初学习率
change_step = 1000 #执行多少轮更新一次学习率
#运行了几轮的BATCH_SIZE的计数器, 且不可训练
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, change_step, learning_rate_decay, staircase=True)

#正则化损失函数
#loss = tf.reduce_mean(tf.square(y_true - prediction)) + tf.add_n(tf.get_collection('losses'))

## 这里使用的这个方法还做了一个第一维结果行差别的求和，reduction_indices=1，实际这个例子每行只有一个结果,使用 loss = tf.reduce_sum(tf.square(y_true - prediction)) 即可
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - prediction), reduction_indices=[1]))

## 训练的operator，AdamOptimizer反正说是最好的训练器, 训练速率0.01
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


# 开始执行
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)     # 初始化saver，用于保存模型
    init = tf.global_variables_initializer()                           # 初始化全部变量
    sess.run(init)                                                     # 初始化全部变量

    ## 要给模型进行训练的数据，只有placeholder类型的需要传进去数据
    feed_dict_train = {inputX: X_train, y_true: y_train, keep_prob_s: 1}
    for i in range(160000):
        _loss, _ = sess.run([loss, train_op], feed_dict=feed_dict_train)  # 训练，注：loss没有训练，只是走到loss，返回值，走到train_op才会训练
        if i % 1000 == 0:
            print("步数:%d\tloss:%.5f" % (i, _loss))

    # 保存模型
    saver.save(sess=sess, save_path="nn_boston_model001/nn_boston.model", global_step=160000)  

## done, 关闭程序后model就会出现在文件夹中

#global_step = tf.Variable(0, trainable=False)
#init_learning_rate = 0.1
#learning_rate = tf.train.exponential_decay(init_learning_rate, global_step=global_step, decay_steps=10, decay_rate=0.9)
##opt = tf.train.GradientDescentOptimizer(learning_rate)
#opt = tf.train.AdadeltaOptimizer(learning_rate)
#
#add_global = global_step.assign_add(1) #定义一个节点，令global_step加1执行完成
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(learning_rate))
#    
#    for i in range(20):
#        g, rate = sess.run([add_global, learning_rate])
#        
#        print(g, rate)




















