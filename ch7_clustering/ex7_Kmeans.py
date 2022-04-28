import math
from operator import index
from random import shuffle
from turtle import pos
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=400)
import pandas
import time
start=time.time()

data=pandas.read_csv('K-means-ex.data',sep=',')

data=np.array(data)
# print(data)


num=len(data)

SSEs=[]
for k in range(2,9):
    # init
    np.random.shuffle(data)
    C=[]
    idx=np.zeros(num)
    for i in range(k):
        C.append(data[i,:])
        idx[i]=i
    # repeat T times
    SSE=0.0
    for T in range(3000):
            
        # compute idx
        for i in range(k,num):
            dis=np.zeros(k)
            for center in range(k):
                dis[center]=math.sqrt((data[i][0]-data[center][0])**2+(data[i][1]-data[center][1])**2)
            idx[i]=np.argmin(dis)
        
        # update center of C[i]
        x=np.zeros(k)
        y=np.zeros(k)
        numinC=np.zeros(k)
        for i in range(num):
            x[int(idx[i])]=x[int(idx[i])]+data[i][0]
            y[int(idx[i])]=y[int(idx[i])]+data[i][1]
            numinC[int(idx[i])]=numinC[int(idx[i])]+1
        for i in range(k):
            x[i]/=numinC[i]
            y[i]/=numinC[i]
            C[i][0]=x[i]
            C[i][1]=y[i]
        SSE=0.0
        for i in range(num):
            SSE+=(data[i][0]-C[int(idx[i])][0])**2+(data[i][1]-C[int(idx[i])][1])**2
        # if(T%50==0): print(SSE)
    SSEs.append(SSE)
    print(C)
    if(k==3):
        fig, ax=plt.subplots(figsize=(12,8))
        for i in range(num):
            ax.plot(data[i][0],data[i][1],'b.')
        for i in C:
            ax.plot(i[0],i[1],'r.')
        plt.show()
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(range(2,9), SSEs)
# plt.show()