# -*- coding:utf-8 -*-
import numpy as np
seed = 2
def generateds():
	#生成基于seed的随机函数
	rdm = np.random.RandomState(seed)
	#随机返回300行2列的矩阵，代表300组坐标点(x0,x1)作为输入数据集
	X = rdm.randn(300, 2)
	#从这组矩阵中取出满足函数条件的数据赋予1,否则赋予0
	Y_ = [int(x0*x0 + x1*x1 < 2) for (x0, x1) in X ]
	#1赋予red，0赋予blue
	Y_c = [['red' if y else 'blue'] for y in Y_ ]
	#对数据进行变形,-1表示n行
	X = np.vstack(X).reshape(-1, 2)
	Y_ = np.vstack(Y_).reshape(-1, 1)
	return X, Y_, Y_c
