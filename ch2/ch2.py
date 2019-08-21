#!/usr/bin/env python 
# -*- coding:utf-8 -*-
__author__ = 'skywoodsz'
"""逻辑回归练习——猫两分类器"""

import numpy as np
import matplotlib.pyplot as plt # matlab 绘图
import h5py # 数据集库
from lr_utils import load_dataset #导入H5数据集
import scipy # 数学库
from PIL import Image #py图像处理库
from scipy import ndimage

'''数据预处理'''
#
# 1.加载数据集
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

# Example of a picture
# index = 13
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" +
#        classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# 2.获取维度
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# 3.向量化
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

# 4.去中心化
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

'''构建模型'''
# Z = WT * X + B
# Y = A(last) = sigmoid(Z)
# L(A,Y) = -Y * log(A) - (1-Y) * log(1-A)
# J = 1/m * sum(L)

# active函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# 初始化参数
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim,1))
    b = 0
    assert(w.shape == (dim,1)) # 断言防错
    assert(isinstance(b,float) or isinstance(b,int)) # isinstance判断数据类型 == type
    return w,b

# 构建网络块： 前向/反向传播
def propagate(w,b,X,Y):
    m = X.shape[1] # data大小
    # forword
    A = sigmoid(np.dot(w.T,X) + b) # A = sigmoid(z); z = WT * X + b
    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    #backword
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads,cost

# 下降 theta = theta - alpha * theta
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return params,grads,costs

def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T,X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0

    assert(Y_prediction.shape == (1,m))
    return Y_prediction

def model(X_train,Y_train,X_test,Y_test,num_iteraions = 2000,learning_rate = 0.5,print_cost = False):
    m = X_train.shape[0] # 不是训练集的size
    w,b = initialize_with_zeros(m)

    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iteraions,learning_rate,print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iteraions}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iteraions = 2000, learning_rate = 0.005, print_cost = True)

# index = 35
# plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
# plt.show()
# print(d["Y_prediction_test"][0,index])
# if d["Y_prediction_test"][0,index] == 1:
#     print("you predicted that it is a cat")
# else:
#     print("you predicted that it is a non_cat")

# costs = np.squeeze(d['costs'])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.show()

# own picture test
my_image = "taitai.png"
fname = "image/" + my_image
img = np.array(Image.open(fname).convert("RGB"))
plt.imshow(img)
plt.show()

img = img/255
m_img = scipy.misc.imresize(img,size=(num_px,num_px,3))
m_img = m_img.reshape(1,num_px*num_px*3).T

m_predicted_img = predict(d["w"],d["b"],m_img)
print("the m_predicted_img =" + str(int(np.squeeze(m_predicted_img))))
if int(np.squeeze(m_predicted_img)) == 1:
    print("you predicted that it is a cat")
else:
    print("you predicted that it is a non_cat")
