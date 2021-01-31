'''
根据成绩推测是不是被录取
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = 'data' + os.sep + "LogiReg_data.txt"  # os.sep \
# 读取数据  header 有时是类型名  有时不是
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Aditted'])
# print(pdData.head())
# print(pdData.shape)  # (100, 3)
# 积极的,正立
positive = pdData[pdData['Aditted'] == 1]
# 消极的，负立
negative = pdData[pdData['Aditted'] == 0]

fig, ax = plt.subplots(figsize=(10, 5))  # 指定画图大小，figsize 指定画图与： 长宽
# 散点图
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admittened')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitten')
ax.legend()
ax.set_xlabel("Exam 1 score")
ax.set_ylabel("Exam 2 score")
# 明显的边界

# 目标建立分类器， 设定阈值 根据阈值 判断录取结果
'''
要完成的模块
sigmoid: 映射到概率函数
model: 返回预测值
cost: 根据参数计算损失
gradient: 计算每个参数的梯度方向
descent: 进行参数更新
accuracy: 计算精度

'''


# sigmoid: 映射到概率函数
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

nums = np.arange(-10, 10, step=1)
# print(nums)  [-10  -9  -8  -7  -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6   7   8   9]
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(nums, sigmoid(nums), 'r')
# plt.show()


# model: 返回预测值
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))

pdData.insert(0, "Ones", 1)  # 添加一列
orig_data = pdData.as_matrix()  # 将帧转换为其Numpy数组表示形式。

cols = orig_data.shape[1]   # 4列
X = orig_data[:,0:cols-1]
# print("--1--")
# print(X)   # 前三列数据
# print("--2--")
Y = orig_data[:,cols-1:cols]  # 最后一列
# print(Y)
theta = np.zeros([1,3])     # 构造参数： [[0. 0. 0.]]  shape = (1, 3)
# print("--3--")
# print(theta.shape)

# cost: 根据参数计算损失  将对数似然函数去负号， 求平均损失
# x: 数据  y: 标签  theta: 参数   model函数： 计算sigmodel
def cost(x,y,theta):
    left = np.multiply(-y,np.log(model(x,theta)))
    right = np.multiply(1-y,np.log(1-model(x,theta)))
    return np.sum(left-right) /len(x)

cost_val = cost(X,Y,theta)
# print(cost_val)
#
# print("--4--")
# print(X)
# print("--5--")
# print(X[:,1])
# gradient: 计算每个参数的梯度方向
# def gradient(x,y,theta):
#     grad = np.zeros(theta.shape) # 梯度， 占位
#     error = (model(x,theta)-y).ravel()
#     for j in range(len(theta.ravel())):
#         term = np.multiply(error,x[:,j])
#
#         grad[0,j] = np.sum(term)/len(x)
#     return grad
def gradient(X, y, theta):
    # print(X.shape)
    # print(y.shape)
    # print(theta.shape)

    grad = np.zeros(theta.shape)
    # print("grad.shape",grad.shape)  #(1, 3)
    # print("model.shape",model(X,theta).shape) # (100, 1)
    # print(model(X,theta))  # 二维数据
    # print("y.shape",y.shape)  # (100,)

    # print(y)  # 一维数据
    #
    # print("model(X,theta)-y) =\n")
    # print((model(X,theta)-y))
    # print("model(X,theta)-y)shape =\n")
    # print((model(X, theta) - y).shape)

    error = (model(X, theta) - y).ravel() # 返回展平数组。
    # print("error =\n")
    # print(error)
    # print("error.shape =\n")

    # print(error.shape)
    for j in range(len(theta.ravel())):  # for each parmeter
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)

    return grad

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

# def stopCriterion(type,value,threshold):
#
#     if type == STOP_ITER: return value > threshold
#     elif type == STOP_COST: return abs(value[-1]-value[-2] < threshold)
#     elif type == STOP_GRAD: return np.linalg.norm(value) < threshold
def stopCriterion(type, value, threshold):
    print(type)
    print(value)
    print(threshold)
    #设定三种不同的停止策略
    if type == STOP_ITER:        return value > threshold
    elif type == STOP_COST:      return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:      return np.linalg.norm(value) < threshold

# 洗牌， 重塑模型
import numpy.random
# 洗牌
def shuffleDate(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:,0:cols-1]
    Y = data[:,cols-1:]
    return X,Y

import time
#  下降
def descent(data,theta,batchSize,stopType,thresh,alpha):
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  #第0个  batch
    X,Y = shuffleDate(data) # 洗牌
    grad = np.zeros(theta.shape) # 计算梯度
    costs = [cost(X,Y,theta)] # 损失值
    # print("&&&&&&&&")
    # print(X.shape,Y.shape)  # (100, 3) (100, 1) (100, 3) (100,)

    while True:
        grad = gradient(X[k:k+batchSize],Y[k:k+batchSize],theta)  # 计算每个参数的梯度方向
        print("计算每个参数的梯度方向 =>\n")
        print(grad)
        k += batchSize
        if k >= 0:
            k =0
            X,Y = shuffleDate(data) # 重新洗牌
        theta = theta - alpha*grad # 参数更新
        costs.append(cost(X,Y,theta))  # 计算损失    # 损失值
        i += 1

        if stopType == STOP_ITER:  value = i
        elif stopType == STOP_COST: value = costs
        elif stopType == STOP_GRAD: value = grad
        print("停止 =>\n")
        # print(" %s : %s : %s " % )

        if stopCriterion(stopType,value,thresh): break  # 值和阈值  对比

    return theta,i-1,costs,grad,time.time()-init_time

#  参数  1 数据  2 参数  3样本数量  3 停止类型  4 阈值  5 学习率
def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)  ##  梯度下降
    #  下面是画图，数据损失数据
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " 原始数据-学习率: {} - ".format(alpha)
    if batchSize==n: strDescType = "Gradient"
    if batchSize==1:  strDescType = "Stochastic"
    else: strDescType = "小批量 ({})".format(batchSize)
    name += strDescType + " 下降 - 停止: "
    if stopType == STOP_ITER: strStop = "{} 迭代".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - 最终损失: {:03.2f} - 执行: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()

    return theta

#选择的梯度下降方法是基于所有样本的
n=100
# 按照迭代次数 目标函数逐步收敛
# Original 原始数据-学习率: 1e-06 - 小批量 (100) 下降 - 停止: 5000 迭代
# Theta: [[-0.00027127  0.00705232  0.00376711]] - Iter: 5000 - 最终损失: 0.63 - 执行: 2.50s
# tt = runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)
# 根据损失值停止
# runExpe(orig_data,theta,n,STOP_COST,thresh=0.000001,alpha=0.001)
# Original data - learning rate: 0.001 - Gradient descent - Stop: costs change < 1e-06
# Theta: [[-5.13364014  0.04771429  0.04072397]] - Iter: 109901 - Last cost: 0.38 - Duration: 131.50s
runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)