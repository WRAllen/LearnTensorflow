###标准版
import tensorflow as tf

tensor = tf.constant('Hello, world!')
with tf.Session() as sess:
    result = sess.run(tensor)
    print(result)

