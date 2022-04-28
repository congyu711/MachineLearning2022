import math
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=400)
import pandas
import time
start=time.time()


traindata=pandas.read_csv('abalone_train.data',sep=',',names=[i for i in range(9)])

testdata=pandas.read_csv('abalone_test.data',sep=',',names=[i for i in range(9)])
traindata=np.array(traindata)
testdata=np.array(testdata)
for data in traindata:
    if(data[0]=='M'): data[0]=0
    elif(data[0]=='F'): data[0]=1
    else: data[0]=2
for data in testdata:
    if(data[0]=='M'): data[0]=0
    elif(data[0]=='F'): data[0]=1
    else: data[0]=2

num = [9, 120, 3]
v = np.random.rand(num[0], num[1])/100
w = np.random.rand(num[1], num[2])/100

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


eta_w = 0.08
eta_v = 0.08
num_train = len(traindata)
totcost = []
N = 30
for t in range(N):
    c = 0.0
    for ii in range(num_train):
        x = traindata[ii].copy()
        x[0]=-1.0

        yy=[0 for i in range(3)]
        yy[traindata[ii][0]]=1

        # a=cost(y,datahw.train_result[ii])
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
        # aa=cost(y,datahw.train_result[ii])
        # print(aa)
        # if(aa>a or aa<0.001):    break
        c += cost(y, yy)
    caloutput()
    totcost.append(c)

# print(w)
correct=0
num_test = len(testdata)
for ii in range(num_test):
    x = testdata[ii].copy()
    x[0]=-1.0
    yy=[0 for i in range(3)]
    yy[testdata[ii][0]]=1
    # init update output (b,y)

    caloutput()
    idx=np.argmax(y)
    if(yy[idx]==1): correct=correct+1

print(correct/num_test)
end=time.time()

print(end-start)


fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(range(N), totcost)
plt.show()


#acc 0.5220994475138122
#time 160.1280312538147