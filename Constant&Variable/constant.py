import tensorflow as tf

#在tensorflow里面申明的数值类型的变量最好要带确切的类型
input1 = tf.constant(2.0, dtype=tf.float32)
input2 = tf.constant(3.0, dtype=tf.float32)
###当然你tf.constant(2.0)申明也不会报错（在该例子里面）

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(ouput))

'''
输出结果：
    6.0
'''