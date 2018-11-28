import tensorflow as tf
import numpy as np

#创建训练模型
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1*x_data + 0.3


#创建测试模型 
Weights = tf.Variable(tf.random_uniform([1]))
biases = tf.Variable(tf.zeros([1]))
#与上面y_data = 0.1*x_data + 0.3相对应
y = Weights*x_data + biases

#计算训练模型与测试模型的误差值
loss = tf.reduce_mean(tf.square(y-y_data))

#传递误差的工作就教给optimizer了, 我们使用的误差传递方法是梯度下降法: Gradient Descent
#这里的0.5是学习速率，越大代表下降的程度越大，不可以过大最好小于１
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#以上只是建立了神经网络的结构，接下来要开始训练了

#初始化变量
init = tf.global_variables_initializer()  

with tf.Session() as sess:
    sess.run(init)
    for step in range(150):
        #开始训练
        sess.run(train)
        if step % 10 == 0:
            print(step, sess.run(Weights), sess.run(biases))
    print("finish learn\n Weights:",sess.run(Weights),"biases:",sess.run(biases))


'''
输出结果：
[0.18697968] [0.25456426]
20 [0.1439452] [0.27704424]
30 [0.12220266] [0.28840193]
40 [0.11121759] [0.29414025]
50 [0.10566752] [0.29703945]
60 [0.10286345] [0.29850423]
70 [0.10144673] [0.29924428]
80 [0.10073093] [0.29961818]
90 [0.10036929] [0.2998071]
100 [0.10018658] [0.29990256]
110 [0.10009427] [0.29995078]
120 [0.10004763] [0.29997513]
130 [0.10002404] [0.29998747]
140 [0.10001215] [0.29999366]
finish learn
 Weights: [0.10000659] biases: [0.29999658]
可以看到Weights和biases都已经十分接近训练模型了，十分令人激动
'''