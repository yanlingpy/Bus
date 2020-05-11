import numpy as np

#多元线性回归
class linearRegression:
    def __init__(self,data_X,learningRate,loopNum):#初始化变量
        # W的shape取决于特征个数，而x的行是样本个数，x的列是特征值个数
        # 所需要的W的形式为 行=特征个数，列=1 这样的矩阵。但也可以用1行，再进行转置：W.T
        # X.shape[0]取X的行数，X.shape[1]取X的列数
        self.W = np.zeros(shape=[1, data_X.shape[1]])#1*n矩阵
        self.b = 0#截距
        self.loopNum = loopNum#迭代次数
        self.learningRate = learningRate#学习率（步长）
    #训练模型
    def error_rate(self,cost,data_Y):
        #计算错误率
        error = 0
        for i in range(len(data_Y)):
            error = abs(data_Y[i] - cost[i])/abs(data_Y[i]) + error
        error_rate_value = error/len(data_Y)
        return error_rate_value

    def fit(self,data_X,data_Y):
        #梯度下降 ：Mini-batch小批量梯度下降算法（批量与随机的综合）
        data_Y = np.array([[i] for i in data_Y])#array数组
        for i in range(self.loopNum):
            #设置w值  与 特征数量相等 
            W_derivative = np.zeros(shape=[1, data_X.shape[1]])#1*n
            b_derivative = 0
            #拟合函数 预测值  # np.dot:矩阵乘法
            WXPlusb = np.dot(data_X, self.W.T) + self.b # W.T：W的转置 (m*n) * (n*1）= m*1
            #w偏导矩阵 计算差值：预测值 - 实际值
            W_derivative += np.dot((WXPlusb - data_Y).T, data_X) #(1*m)*(m*n)=1*n
            #更新截距
            b_derivative += np.dot(np.ones(shape=[1, data_X.shape[0]]), WXPlusb - data_Y)#(1*m)*(m*1)=1
            #计算W 梯度
            W_derivative = W_derivative / data_X.shape[0] # data_X.shape[0]:data_X矩阵的行数，即样本个数
            b_derivative = b_derivative / data_X.shape[0]
            #反向更新参数 
            self.W = self.W - self.learningRate*W_derivative
            self.b = self.b - self.learningRate*b_derivative
            if i % 100 == 0:
                print("错误率:" + str(self.error_rate(WXPlusb,data_Y)))
    def predict(self, X):
        result_list = []
        for i in range(X.shape[0]):
            aa = np.dot(X[i], self.W[0]) +  self.b[0][0] 
            result_list.append(aa)
        return result_list       
  