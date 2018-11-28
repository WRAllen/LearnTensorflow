###不需要通过Session来运行
import tensorflow as tf
import tensorflow.contrib.eager as tfe 
###引入了动态图机制Eager Execution
tfe.enable_eager_execution() 
###申明一个常量
tensor = tf.constant('Hello, world!(no need session)')
# print(tensor)
tensor_value = tensor.numpy()
print(tensor_value)