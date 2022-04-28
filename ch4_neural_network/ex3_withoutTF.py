#   3-layer neural network with sigmoid function

import math
import data
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=400)


num = [64, 63, 10]
v = np.random.rand(num[0], num[1])/100
w = np.random.rand(num[1], num[2])/100
# print(data.train_data)

# output
x = np.zeros(num[0])
b = np.zeros(num[1])
y = np.zeros(num[2])
b[0] = -1.0
x[0] = -1.0

# print(x)


def f(x):   # sigmoid
    return 1.0/(1.0+math.exp(-x))


def df(x):  # diff of sigmoid
    tmp = f(x)
    return tmp*(1-tmp)


def cost(y, resy):
    res = 0.0
    for i in range(num[2]):
        res += (y[i]-resy[i])**2
    res *= 0.5
    return res


def caloutput():
    for i in range(1, num[1]):
        b[i] = 0
        for j in range(num[0]):
            b[i] += v[j, i]*x[j]
        b[i] = f(b[i])

    for i in range(0, num[2]):
        y[i] = 0
        for j in range(num[1]):
            y[i] += w[j, i]*b[j]
        y[i] = f(y[i])


eta_w = 0.1
eta_v = 0.1
num_train = len(data.train_data)
totcost = []
N = 2000
for t in range(N):
    c = 0.0
    for ii in range(num_train):
        data_x = data.train_data[ii]
        yy = data.train_result[ii]
        for i in range(1, 64):
            x[i] = data_x[i-1]

        # a=cost(y,data.train_result[ii])
        caloutput()
        # print(y)
        # input()
        g = np.empty((num[2]))
        for j in range(num[2]):
            g[j] = y[j]*(1-y[j])*(yy[j]-y[j])
        

        # train v[i,h]
        for h in range(0, num[1]):
            ee = 0.0
            for j in range(num[2]):
                ee += g[j]*w[h, j]
            eh = ee*b[h]*(1.0-b[h])
            for i in range(num[0]):
                v[i, h] += eta_v*eh*x[i]
        # train w[h,j]
        for j in range(num[2]):
            for h in range(num[1]):
                w[h, j] += eta_w*g[j]*b[h]
        # aa=cost(y,data.train_result[ii])
        # print(aa)
        # if(aa>a or aa<0.001):    break
        c += cost(y, data.train_result[ii])
    caloutput()
    totcost.append(c)

# print(w)
num_test = len(data.test_data)
for ii in range(num_test):
    data_x = data.test_data[ii]
    yy = data.test_result[ii]
    for i in range(1, 64):
        x[i] = data_x[i-1]
    # init update output (b,y)

    caloutput()

    print(y)
    print(cost(y, data.test_result[ii]))
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(range(N), totcost)
plt.show()
