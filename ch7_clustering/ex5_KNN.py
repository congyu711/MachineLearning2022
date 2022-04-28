import math
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=400)
import time
start=time.time()

K=10
data=np.empty((20,5))

with open('KNN.data') as file:
    strs = file.readlines()
    T=0
    for line in strs:
        tmp=line.split(' ')
        # print(tmp)
        for i in range(5):
            data[T][i]=float(tmp[i])
        T=T+1

# print(data)
np.random.shuffle(data)
print(data)
# data[0-15] train data
# data[16-19] test data


corr=0
for i in range(16,20):
    dist=np.empty((16,1))
    for j in range(16):
        dis=0.
        for k in range(4):
            dis+=(data[i][k]-data[j][k])**2
        dis=math.sqrt(dis)
        dist[j][0]=dis
    # print(dist)
    idx=np.argsort(dist[:,0])
    num0=0
    for ii in range(10):
        if(data[idx[ii]][4]==0): num0=num0+1
    print("num0:",num0)
    print("data[i][4]: ",data[i][4])
    cat=0
    if(num0>=5): cat=0
    else: cat=1
    if(cat==data[i][4]): corr=corr+1
print(corr/4*100,'%')
