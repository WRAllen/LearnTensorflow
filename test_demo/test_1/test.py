#conding:utf-8
import tensorflow as tf
import pandas as pd
import os
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



def train_full(datas):
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

    loss = tf.reduce_mean(tf.square(layer_out-Y))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

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
               _, loss_value = sess.run([train_step, loss], feed_dict={X:datas.values[step:step+BATCH_SIZE,2:].tolist(), Y:datas.values[step:step+BATCH_SIZE,1].reshape([10,1])}) 
            if i % 500 == 0:
                 print("经过%d轮训练，loss是：%f"%(i, loss_value))
                 saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))





datas = pd.read_csv("data.csv")
print("开始训练")
train_full(datas)
print("训练结束")

