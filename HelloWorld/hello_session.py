###标准版
import tensorflow as tf
###申明一个常量
tensor = tf.constant('Hello, world!')
# print(tensor)
with tf.Session() as sess:
    result = sess.run(tensor)
    print(result)

