import time
import pandas
import math
import random
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=400)
start = time.time()

data = pandas.read_csv('adaboost.data', sep=' ')
data = np.array(data)
random.seed(time.time())
# print(data)

# one-level decision tree.
# use random decision varible.
weakclassifier = []

num_classifier = 11
N=48
d = np.zeros(N)
for i in range(N):
    d[i] = 1/N
a = np.zeros(num_classifier)
for i in range(num_classifier):
    a[i] = 1/num_classifier

for i in range(num_classifier):
    cl = []
    cl.append(random.randint(0, 1))
    # print("cl[0] ",cl[0])
    # index = np.argsort(data[:, cl[0]])
    maxidx = random.randint(0, 20)
    maxcorr = 0.
    for p in range(N):
        if(data[p][cl[0]] < data[maxidx][cl[0]] and data[p][2] == -1):
            maxcorr = maxcorr+d[p]
        if(data[p][cl[0]] >= data[maxidx][cl[0]] and data[p][2] == 1):
            maxcorr = maxcorr+d[p]

    for idx in range(N):
        corr = 0.
        for p in range(N):
            if(data[p][cl[0]] < data[idx][cl[0]] and data[p][2] == -1):
                corr = corr+d[p]
            if(data[p][cl[0]] >= data[idx][cl[0]] and data[p][2] == 1):
                corr = corr+d[p]
        if(maxcorr <= corr ):
            maxidx = idx
            maxcorr = corr
        print(corr)
    print("maxcorr",maxcorr)
    cl.append(data[maxidx][cl[0]])
    a[i] = 0.5*math.log(maxcorr/(1-maxcorr))
    weakclassifier.append(cl)
    totald = 0.
    for j in range(N):
        if((data[j][cl[0]] < data[idx][cl[0]] and data[j][2] == -1) or
                (data[j][cl[0]] >= data[idx][cl[0]] and data[j][2] == 1)):
            d[j] = d[j]*math.exp(-a[i])
        else:
            d[j] = d[j]*math.exp(a[i])
        totald = totald+d[j]
    for j in range(N):
        d[j] /= (totald+0.01)


print("weakclassifier: ", weakclassifier)
# print(d)
# print(a)
fig, ax=plt.subplots(figsize=(12,8))
correct=0
for dd in data:
    dec = 0.0
    for i in range(num_classifier):
        if(dd[weakclassifier[i][0]] < weakclassifier[i][1]):
            dec-=a[i]
        elif(dd[weakclassifier[i][0]] >= weakclassifier[i][1]):
            dec+=a[i]
    # print(dd,dec)
    if(dec>0):  ax.plot(dd[0],dd[1],'r.')
    else: ax.plot(dd[0],dd[1],'b.')
    if(dd[2]==-1): plt.scatter(dd[0], dd[1], c='b', marker='o', edgecolors='b',alpha=0.5)
    else: plt.scatter(dd[0], dd[1], c='r', marker='o', edgecolors='r',alpha=0.5)
    if((dec>0 and dd[2]==1) or (dec<=0 and dd[2]==-1)): correct=correct+1

print(correct, correct/N)
plt.show()
# 1
# maxcorr 0.8541666666666671
# weakclassifier:  [[1, 69]]
# 41 0.8541666666666666


# 2
# maxcorr 0.7644726709906667
# weakclassifier:  [[1, 69], [0, 170]]
# 41 0.8541666666666666


# 11
# maxcorr 0.7960035782693435
# weakclassifier:  [[1, 69], [0, 170], [1, 65], [0, 170], [1, 65], [0, 170], [1, 65], [1, 65], [0, 170], [1, 65], [1, 65]]        
# 40 0.8333333333333334

