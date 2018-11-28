import tensorflow as tf

input1 = tf.Variable([3])
#要更改变量的值，请使用 assign 操作：（向变量赋予新值时，其形状必须和之前的形状一致：）
tf.assign(input1, [7])

###变量申明后记得一定要初始化，不然不能运行
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(input1))

'''
输出结果：
    [3]
'''