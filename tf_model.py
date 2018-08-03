# coding: utf-8
import tensorflow as tf
import numpy as np

"""
怎么理解？
对于某个现象，想用一个模型来刻画它
y = W * x + b
W，b是参数
x是输入，y是输出
tf使用梯度下降法进行拟合
一开始tf认为W=[[-0.94402915  0.8266388 ]] b=[1.3738805]就能拟合
然后发现有误差，所以进行了修正，下一次使用W=[[-0.26090068  0.19071952]]  b=[0.5123181]，还是有误差
在两百次后使用[[0.09997699 0.19998224]] [0.30002204]拟合
这是一个误差越来越小的过程

tf.random_uniform([1, 2], -2.0, 1.0)和tf.zeros([1])是对W和b的范围预估

"""

if __name__ == "__main__":

	# 1.准备数据：使用 NumPy 生成假数据(phony data), 总共 100 个点.
	x_data = np.float32(np.random.rand(2, 100)) # 随机输入
	y_data = np.dot([0.100, 0.200], x_data) + 0.300

	# 2.构造一个线性模型
	b = tf.Variable(tf.zeros([1]))
	W = tf.Variable(tf.random_uniform([1, 2], -2.0, 1.0))
	y = tf.matmul(W, x_data) + b

	# 3.求解模型
	# 设置损失函数：误差的均方差
	loss = tf.reduce_mean(tf.square(y - y_data))
	# 选择梯度下降的方法
	optimizer = tf.train.GradientDescentOptimizer(0.5)
	# 迭代的目标：最小化损失函数
	train = optimizer.minimize(loss)


	############################################################
	# 以下是用 tf 来解决上面的任务
	# 1.初始化变量：tf 的必备步骤，主要声明了变量，就必须初始化才能用
	init = tf.global_variables_initializer()


	# 设置tensorflow对GPU的使用按需分配
	config  = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	# 2.启动图 (graph)
	sess = tf.Session(config=config)
	sess.run(init)

	# 3.迭代，反复执行上面的最小化损失函数这一操作（train op）,拟合平面
	for step in range(0, 201):
		t = sess.run(train)
		if step % 20 == 0:
			print(t)
			print(step, sess.run(W), sess.run(b))

	# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
	pass

