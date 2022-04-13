- [link](https://github.com/dragen1860/Deep-Learning-with-TensorFlow-book)
### 问题
- 为什么要batch计算？这个其实需要调试，一般从128开始选
- 为什么需要多个hidden layer？
    - 在神经网络中，当且仅当数据非线性分离时才需要隐藏层！
- 激活函数，y=f(x)，找到一个f能够将x映射成y
- 损失函数，表示训练数据和模型的gap大小
- 损失函数越小越好？越小越拟合，但也可能出现过拟合
- gradient clipping 改变梯度下降的模，从而更好的寻参
- 为什么用sigmoid函数进行压缩呢？
- 信息论中的信息熵公式 sum(-Pi*log(Pi)) 最小为0表示掌握了所有的信息，熵越大表示越不确定
- 传播算法 
    - [公式推导](https://www.youtube.com/watch?v=rq-UMTcI6Tk&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=68)
    - 前向传播算法 Forward propagation，正常的按每一层计算？
    - 反向传播算法 Back propagation，直接求导最后的E和第一层的w的关系？
    - Perceptron（感知机），激活函数使用 sigmoid函数 softmax函数
        - 输入层X(0,0)->中间层X(1,0)->输出层O(1,0)->损失函数E<-真实值t(0)
        - 输入层到中间层经过W(1,0,0)
        - logistic中间层到输出层经过激活函数sigmoid(x)=1/(1+exp(-x))
        - d(sigmoid(x)) = sigmoid(x)(1-sigmoid(x))dx
        - 对X向量求softmax，p(i)=softmax(i)=exp(X(i))/(sum(X(k) for k in range len(X)))
        - dp(i)/dx(j)=p(i)/(1-p(j))
        - dX(1,0)/dW(1,0,0)=X(0,0)
        - dE/dW(1,0,0)=O(1,0)(1-O(1,0)) * X(0,0) * (O(1,0)-t(0))
        - 当推导出单层感知机后可以通过链式法则推广到多层感知机
        - 函数优化：求极小值
- 为什么mnist通过几次线性拟合就能完成预测？
    - 实际上是一种特征提取和降维，将28*28 784维的数据降维成256，再降维成128再降维成10
    - 那为什么不用更加一般化的泰勒展开多项式呢？
- 为什么叫dense？应为全连接所以密集
- 拟合优化，减少过拟合(over fitting)
    - regularization，添加项的方式优化损失函数
    - momentum，通过考虑之前的梯度更新的优化learning_rate，达到减少尖锐的更新梯度方向和跨越一些不是最优的极值点的目的
    - lr tuning，动态调整learning rate，例如连续十次loss没有减少的时候，减小lr试探
    - early stop，当test set无法下降的时候，以此时的loss值为准
    - [dropout]https://www.youtube.com/watch?v=yFfZEDjVmCc&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=95，通过随机断掉某个链接的方式，本质是越少的参数训练越好，但是在test的时候是全部用上的
- [卷积有什么用](https://www.youtube.com/watch?v=_FZ591_x2sY&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=97)
    - 可以抽象像素概念到搞成特征概念，例如识别车子，就需要识别四个轮子
- feature scaling，特征缩放，以0为均值，1为方差，(x-均值)/方差 BatchNormalization
- ResNet，通过shortcut使得网络能够兼顾多层的参数量训练和退化到之前的版本的能力
- fc层，full connection全连接层
- 通过机器学习解决问题的[方案](https://www.youtube.com/watch?v=9rGDGuZ_erQ&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=116)
- 无监督学习的目标是重建自己
- 人是如何思考和学习的？能否把思考和学习的能力通过计算机实现？
- rnn/lstm/gru的区别是啥？rnn是解决序列化训练问题，但是优化容易出现梯度爆炸和梯度离散所以使用lstm，gru在lstm的基础上增加了几个门
- 如何理解数学公式，https://www.youtube.com/watch?v=ZRJR19E6it0&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=133

- 协同过滤
    - 基于物品的协同过滤（物以类聚）
    sim(u, s) = sum(sim(si, s) * score(u, si) for si in S)
    sim(si, s) 通过用户对物品的反馈表计算得到，每个用户对物品的反馈组成向量，两个向量的余弦值即为相似度
    - 基于用户的协同过滤（人以群分）
    sim(u, s) = sum(sim(u, ui) * score(ui, s) for ui in U)
    sim(u, ui) 和sim(si, s)计算类似
- 召回率RECALL=TP/(TP+FN)、精确率PRECISION=TP/(TP+FP)、准确率ACCURACY=(TP+TN)/(TP+FN+FP+TN)
- 对于100个样本，其中60个正样本，40个负样本
    - 如果系统A预测50个样本为正，但只有40个是预测正确的，TP=40,FP=10,TN=30,FN=20
        - RECALL=67%, PRECISION=80%, ACCURACY=70%
    - 如果系统B预测100个样本为正，TP=60,FP=40,TN=0,FN=0
        - RECALL=100%, PRECISION=60%, ACCURACY=60%
    - 如果系统B预测100个样本为负，TP=0,FP=0,TN=40,FN=60
        - RECALL=0%, PRECISION=0%, ACCURACY=40%
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    