# -*- coding:utf-8 -*-
#详细的交叉熵的使用可以看之前的demo
#cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0))+(1-y)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
# 接下来介绍一下交叉熵里面的相关函数
#这个函数可以将一个张量中的数值限定在一个范围之内例如：1e-10, 1.0，小于1e-10(0.00000000001)的会变成1e-10,大于１的会变成1,避免无效的值
#tf.clip_by_value(1-y, 1e-10, 1.0)
#这是个取对数的函数
#tf.log(tf.constant([1.0, 2.0, 3.0]))
#求平均值
#tf.reduce_mean()
