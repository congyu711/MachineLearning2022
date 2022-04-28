from cmath import e, exp, log
import math
import numpy as np
import random
import matplotlib.pyplot as plt

trN=1300
tsN=300
N=1600
x1=np.ones((N//2,3))
x2=np.ones((N//2,3))
# ys=np.empty((trN,1))
# cat=[]
xx=[]
yy=[]

for i in range(800):
    x1[i,0]=np.random.normal(1.)
    x1[i,1]=np.random.normal(1.)
    x1[i,2]=1.

for i in range(800):
    x2[i,0]=np.random.normal(2.5)
    x2[i,1]=np.random.normal(2.5)
    x2[i,2]=0.

# with open('iris_train.data') as file:
#     T=0
#     for line1 in file:
#         tmp=line1.split(',')
#         for i in range(4):
#             xs[T][i+1]=float(tmp[i])
#         T=T+1
#         cat.append(tmp[4])
xs=np.ones((trN,3))
ys=np.ones((trN,1))
xt=np.ones((tsN,3))
yt=np.ones((tsN,1))

random.shuffle(x1)
random.shuffle(x2)

for i in range(650):
    xs[i,1]=x1[i,0]
    xs[i+65,1]=x2[i,0]
    xs[i,2]=x1[i,1]
    xs[i+65,2]=x2[i,1]
    ys[i,0]=x1[i,2]
    ys[i+65,0]=x2[i,2]

for i in range(150):
    xt[i,1]=x1[i+65,0]
    xt[i+15,1]=x2[i+65,0]
    xt[i,2]=x1[i+65,1]
    xt[i+15,2]=x2[i+65,1]
    yt[i,0]=x1[i+65,2]
    yt[i+15,0]=x2[i+65,2]

# beta=np.array([[5.21],[0.86],[-3.35],[0.34],[-1.842]])
beta=np.ones((3,1))
# for i in range(trN):
#     if(cat[i]=='Iris-versicolor\n'): y[i,0]=1
#     else: y[i,0]=0

# print(xs)

def cost(beta,xs,y):
    res=0.0
    for i in range(trN):
        bx=xs[i,:].dot(beta)
        res-=y[i,0]*bx
        res+=math.log(1+math.exp(bx))
    return res

def _argmin(beta):
    N=5000
    alpha=0.0017
    # tmp=beta
    for k in range(N):
        # alpha=random()/500
        cs=cost(beta,xs,ys)
        gd=np.zeros((3,1))
        if(k%50==0): 
            print(k,"====",cs)
            xx.append(k)
            yy.append(cs)
        for i in range(trN):
            xi=xs[i,:].reshape((1,3)).T
            gd-=ys[i,0]*xi
            exb=math.exp(xi.T.dot(beta))
            gd+=xi*exb/(1+exb)
        # print(gd)
        if(cost(beta-gd*alpha,xs,ys)>cs):
            print(beta)
            break
        beta-=gd*alpha
    return beta

beta=_argmin(beta)
print(beta)

# 49900 ==== [0.22792404]
# [[17.90966966]
#  [-3.74804834]
#  [-2.16643522]]

correct=0
for k in range(tsN):
    exb = xs[k, :].dot(beta)
    p = 1.0/(1+math.exp(-exb))
    if(p>0.5 and ys[k,0]==1 or p<0.5 and ys[k,0]==0): correct=correct+1

print(correct,tsN,correct/tsN)


plt.subplot(1,2,1)

plt.plot(xs[:,1],xs[:,2],'b.')
plt.plot(x2[:,0],x2[:,1],'r.')
bx=plt.subplot(1,2,2)
plt.plot(xx,yy,'-')

plt.show()
