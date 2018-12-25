# -*- coding:utf-8 -*-
import tensorflow as tf

input1 = tf.Variable([2])
input2 = tf.Variable([6])
#要更改变量的值，请使用 assign 操作：（向变量赋予新值时，其形状必须和之前的形状一致,如果不一致要加上validate_shape=False）
tf.assign(input1, input2)
tf.assign(input1, [7,1], validate_shape=False)
###上面这2个并没有改变input1的值，因为没有运行
input1.assign(input2)
#上面这个会报错，因为形状不一样
#input1.assign(input2, validate_shape=False)没有这种写法


'''
#随机初始化tf的变量
#几种随机数生成函数
#tf.random_normal： 随机数符合正太分布　主要参数：平均值，标准差，取值类型
#tf.truncated_normal：同上就是如果随机出来的值偏离平均值超过2个标准差就会重新随机　主要参数：平均值，标准差，取值类型
#tf.random_uniform：均匀分布　主要参数：最小，最大取值，取值类型
#tf.random_gamma：Gamma分布　主要参数：形状参数alpha，尺度参数beta，取值类型
'''
var_1 = tf.Variable(tf.random_normal([2,3], stddev=2))

###变量申明后记得一定要初始化，不然不能运行
init = tf.global_variables_initializer()
#with会自动调用close()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(input1))
    print("下面是调入运行结果")
    print(sess.run(tf.assign(input1, input2)))
    print(sess.run(tf.assign(input1, [7,1], validate_shape=False)))
    # print(sess.run(input1.assign(input2)))
    # print(sess.run(input1.assign(input2, validate_shape=False)))
    print("下面是变量初始化结果")
    print(sess.run(var_1))

'''
输出结果：
[2]
下面是调入运行结果
[6]
[7 1]
下面是变量初始化结果
[[ 0.79270667  0.90415007  3.752612  ]
 [-1.3249805   1.7869377   1.0594661 ]]

'''



	