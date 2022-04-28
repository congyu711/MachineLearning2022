from cmath import e, exp, log
import math
import numpy as np


trN=59
tsN=32
xs=np.ones((trN,5))
ys=np.empty((trN,1))
cat=[]


with open('iris_train.data') as file:
    T=0
    for line1 in file:
        tmp=line1.split(',')
        for i in range(4):
            xs[T][i+1]=float(tmp[i])
        T=T+1
        cat.append(tmp[4])
    

beta=np.array([[5.21],[0.86],[-3.35],[0.34],[-1.842]])
# print(beta)
y=np.empty((trN,1))
for i in range(trN):
    if(cat[i]=='Iris-versicolor\n'): y[i,0]=1
    else: y[i,0]=0

# print(y)

def cost(beta,xs,y):
    res=0.0
    for i in range(trN):
        bx=xs[i,:].dot(beta)
        res-=y[i,0]*bx
        res+=math.log(1+math.exp(bx))
    return res

def _argmin(beta):
    N=50000
    alpha=0.001
    # tmp=beta
    for k in range(N):
        # alpha=random()/500
        cs=cost(beta,xs,y)
        gd=np.zeros((5,1))
        if(k%100==0): print(cs)
        for i in range(trN):
            xi=xs[i,:].reshape((1,5)).T
            gd-=y[i,0]*xi
            exb=math.exp(xi.T.dot(beta))
            gd+=xi*exb/(1+exb)
        # print(gd)
        if(cost(beta-gd*alpha,xs,y)>cs):
            print(beta)
            break
        beta-=gd*alpha
    return beta

beta=_argmin(beta)
print(beta)