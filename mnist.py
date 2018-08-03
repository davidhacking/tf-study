# coding=utf-8
import tensorflow as tf
from utils.logger import Logger
import numpy
from PIL import Image
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=numpy.nan)

"""
Y = X * W + B
Xij表示第i张图片的第j个像素是多少
Wij表示图像中像素i的数值对于这张图片是数字j能够贡献多少的权重
T=X*W Tij表示第i中图片是数字j的概率是多少

softmax
https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0
归一化指数函数 -- 梯度对数归一化
array = [...]
array_exp = [math.exp(i) for i in z]
sum_array_exp = sum(array_exp)
softmax_array = [i/sum_array_exp for i in array_exp]
原来softmax是和max相对的概念，max(a, b)表示a和b中的大者，max(array)表示array中的最大数
softmax表示以一定的概率取一个数，大的数取的概率也更大

softmax loss
cross-entropy 交叉熵
y_ * log(y)
y表示的是机器学习过程中认为的图片的数字的概率分布
y_是实际的概率分布
log(yi)是单个图片是某个数字的信息熵
y_ * log(y)是整体的信息熵
我的理解是这里主要要找到一个数值衡量y_ 和y的差距有多大，而y_ * log(y)恰好能提供这样的衡量

mnist数据集是一个格式化的byte文件，通过format可以重新变成序列化对象

梯度
对于一个函数f(x,y), grad f(x,y)=(df/dx, df/dy)，对于函数f(x,y)而言都是斜率，可以知道在某一点处函数的单调性
步长
为了求f(x,y)的极小值或极大值，例如给出了在x0处的grad f(x0,y0)，那么下一个应该选的点就是x1=x0-n*grad f(x0,y0)

MNIST模型的理解
MNIST模型假设任何一张输入的图片以W*x+b的概率分布对应0-9这几个数字
一开始W和b都是0向量，这样输出表示这张图片不表示任何数字，这时使用已经标记的数据就行修正，修正W和b
采用的方法就是梯度下降方法



reduce_sum
包含reduce和sum两个操作，sum就是求和，reduce实际是降维
x = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])
array([[[1, 2],
        [3, 4]],

       [[5, 6],
        [7, 8]]])
sess.run(tf.reduce_sum(x)) # 36 对所有数求和
sess.run(tf.reduce_sum(x, axis=1)) 
array([[ 4,  6],
       [12, 14]]) 对第一维进行求和（从零开始计数）
       
       
熵
消息的熵 * 消息的长度 决定了 消息能携带多少信息
熵越大，能够携带的信息量就越多
y=xlogx 当x趋向于0时，y趋向于0  因为 logx/(1/x) 1/x在x趋向于0的过程中趋向于正无穷的速度大于logx（我的数学智商还在，可怕）

机器学习中常用的优化方法是使熵增大
http://www.shareditor.com/blogshow?blogId=98
熵的定义应该是在宏观上是可加的，在微观上是可乘的
一个随机变量ξ有A1、A2、A3……共n个不同的结果，每个结果出现的概率是p1、p2、p3……
那么我们把ξ的不确定度定义为信息熵，参考上面物理学熵的定义，A1、A2、A3……可以理解为不同的微观状态，
那么看起来信息熵应该是log n喽？不然，因为这个随机变量ξ一次只能取一个值而不是多个值，
所以应该按概率把ξ劈开，劈成n份，每份的微观状态数分别是1/p1、1/p2、1/p3……，
这样这n份的熵分别是log 1/p1、log 1/p2、log 1/p3……，
再根据熵的可加性原理，得到整体随机变量ξ的信息熵是∑(p log 1/p)，即H(ξ) = -∑(p log p)

H(Y|X) = -∑ p(x)p(y|x) log p(y|x)


GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

Optimizer
使用方法
初始化一个opt
opt = XXXOptimizer
设置调整参数
opt.minimize(cost, var_list)
开始优化
opt.run()

如果想自己处理minimize过程中的梯度
可以将每一步变成三步走
grads_and_vars = opt.compute_gradients(loss, <list of variables>)
# process gradients
capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]
opt.apply_gradients(capped_grads_and_vars)

自己写的优化器可以使用slot这个东西进行调试


梯度下降
-数学知识
	y=f(x)的二阶导数为什么即为d^2y/dx^2？其实以前就思考过了，这里想再次刨根究底一发
	二阶导数实际上是对dy/dx进行求导得到：(ddy * dx - dy * ddx)/(dx)^3，这里ddx是相对于x的高阶无穷小
	所以可以省略，所以最后是ddy/(dx)^2最后即为d^2y/dx^2。那么ddy也是相对于y的无穷小，为啥不是零？
	因为当dx->0时，ddy不一定->0，可以想象dy/dx是一个在x0处函数值变化极大的函数，这时ddy可能相对于dx很大很大。
	而为什么ddx是相对于x的高阶无穷小呢？
	假设有x，x+a，x+b三个数，0<a,b a,b->0，a,b两个是同阶无穷小，那么deltaX1=a, deltaX2=b，delta delta X=a-b
	那么可以认为delta delta X相对于x来说是可以忽略的了
	
	对矩阵求导数，就是对矩阵求梯度
	
概率论
	贝叶斯
	假设有个厂要招人，招的工人可能吸毒的概率是0.5%，现在想知道这些人到底有没有吸毒，所以去医院检查。
	医院说如果一个人真吸了毒我们检查出来是阳性，也就是吸了毒的概率是99%，如果一个人不吸毒检查出来也是不吸毒的概率也是99%。
	那么现在有个人检查出来报告是吸毒的，这个人真的也是吸毒的概率是多少？
	只有33.22%的概率这个人真的吸毒
	img/条件概率.png
	P(AB)=P(A|B)P(B) => P(A|B)=P(AB)/P(B)
	为何条件概率这样定义？
	实际上P(AB)=P(B)P(A|B)这样写就能理解了，就是说A和B同时发生的概率可以先求B发生的概率，再求在B发生的条件下A发生的概率。
	最后将两个概率相乘就是A和B同时发生的概率了。
	从集合的角度和容易理解，如图所示，假设全集为T，数量为1，那么其中的A,B,X刚好对应P(A),P(B),P(AB)，
	P(A|B)=X/B=P(AB)/P(B)=X/B
	全概率公式
	P(B)=P(AB)+P(_AB)=P(B|A)P(A)+P(B|_A)P(_A)
	P(A|B)=P(AB)/P(B)=P(AB)/(P(B|A)P(A)+P(B|_A)P(_A))
	
矩阵
	直观理解
		1. Ax其中A是矩阵，x是列向量，则b=Ax是对于x的变换，b向量是与x在同一向量空间的另一个向量
		2. 也可以理解为A是一个向量空间，例如：I3向量实际是一个三维的向量空间，b=Ax，实际是将
		A中的每个维度的向量重新进行了组合，得到了b向量，b向量也属于三维空间，是由Ax描述得到的
	矩阵的逆，b=Ax将x向量变换成了b向量，A^-1b则将b向量变换回了x向量。这样AA^-1=I，也能直观的理解了
	即将A中的列向量使用A^-1中的列向量进行线性组合，得到了三个维度的向量I
		也可以从右往左去理解，即左边行向量对右边行向量进行了线性组合
	
	矩阵的秩，矩阵的秩描述了矩阵中不共线的(行|列)向量个数，这意味着能够描述多少维的空间
	行列式，描述的A的所有(行|列)向量所围成的有向体积，所以A是个m*n的矩阵，m!=n则，在m为空间(m>n)中，
	A的体积为0
		https://zh.wikipedia.org/wiki/%E8%A1%8C%E5%88%97%E5%BC%8F
	
	特征值和特征向量，Ax=lambda x，根据定义可以得出X^-1AX=V
		https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E5%80%BC%E5%92%8C%E7%89%B9%E5%BE%81%E5%90%91%E9%87%8F
		
约束条件求极值问题
	KKT https://zhuanlan.zhihu.com/p/26514613
"""


if __name__ == '__main__':
	import tensorflow.examples.tutorials.mnist.input_data as input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# 28 * 28 = 784 10表示0-9
	x = tf.placeholder("float", [None, 784]) # 每张图片784个像素，有n个图
	W = tf.Variable(tf.zeros([784, 10])) # 对一张图片是每个数字的概率
	b = tf.Variable(tf.zeros([10])) # 偏置量 因为输入有差异，用这么个变量代表
	y = tf.nn.softmax(tf.matmul(x, W) + b) # 使用softmax函数进行评估
	y_ = tf.placeholder("float", [None, 10]) # 图片中所写数字的实际值例如是0则y_=[1,0,0,0,0,0,0,0,0,0]
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
	# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	opt = tf.train.GradientDescentOptimizer(0.01)
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		for i in range(mnist.train.num_examples):
			batch_xs, batch_ys = mnist.train.next_batch(1)
			Logger.instance().info("i: ", i)
			Logger.instance().info("before: ")
			Logger.instance().info("W=", sess.run(W))
			Logger.instance().info("b=", sess.run(b))
			Logger.instance().info("x=", batch_xs)
			y_value = y.eval({x: batch_xs})
			Logger.instance().info("y=", y_value)
			Logger.instance().info("y_=", batch_ys)
			cross_entropy_value = cross_entropy.eval({y: y_value, y_: batch_ys})
			Logger.instance().info("cross_entropy=", cross_entropy_value)
			# sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
			# (gradient, variable)
			grads_and_vars = sess.run(opt.compute_gradients(cross_entropy), feed_dict={x: batch_xs, y_: batch_ys})
			sess.run(opt.apply_gradients(opt.compute_gradients(cross_entropy)), feed_dict={x: batch_xs, y_: batch_ys})
			Logger.instance().info("after: ")
			Logger.instance().info("W=", sess.run(W))
			Logger.instance().info("b=", sess.run(b))
			Logger.instance().info()

		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) #0.9164
