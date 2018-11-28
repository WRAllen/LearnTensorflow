###不需要通过Session来运行
import tensorflow as tf
import tensorflow.contrib.eager as tfe 
###引入了动态图机制Eager Execution
tfe.enable_eager_execution() 

tensor = tf.constant('Hello, world!(no need session)')
tensor_value = tensor.numpy()
print(tensor_value)