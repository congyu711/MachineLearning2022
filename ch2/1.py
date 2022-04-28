import numpy as np
import matplotlib.pyplot as plt

N=int(3e3)
W=7.0
B=-2.0
x=np.random.rand(N,1)
y=W*x+B+(np.random.randn(N,1)-0.5)

# ordinary least squares

sumxy=0.
sumx=0.
sumy=0.
sumx2=0.
for i in range(0,N):
    sumxy+=(x[i]*y[i])
    sumx+=x[i]
    sumy+=y[i]
    sumx2+=(x[i]**2)

ols_w=(sumxy-sumx*sumy/N)/(sumx2-sumx**2/N)

ols_b=sumy/N-sumx/N*ols_w

# print(ols_w)
# print(ols_b)

# gradient_descent

gd_w=0.
gd_b=0.

epochs=1e3
lr=0.05

for T in range(int(epochs)):
    t1=1
    t2=1
    for i in range(N):
        t1+= (gd_w*x[i]+gd_b-y[i])*x[i]
        t2+= (gd_w*x[i]+gd_b-y[i])
    gd_w-=lr*t1/N
    gd_b-=lr*t2/N



fig, ax=plt.subplots(figsize=(12,8))

ax.plot(x,y,'b.')
ax.plot(x,ols_w*x+ols_b,'r-')
ax.plot(x,gd_w*x+gd_b,'g-')
plt.show()