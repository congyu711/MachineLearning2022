import numpy as np
import xlrd
import math

table=xlrd.open_workbook('cement.xlsx').sheet_by_name('Sheet1')

# for col in range(table.ncols):
#     for row in range(table.nrows):
#         print(table.cell_value(row,col))


xs=np.empty((table.nrows-1,table.ncols-2))
ys=np.empty((table.nrows-1,1))

for col in range(2,table.ncols):
    for row in range(1,table.nrows):
        xs[row-1][col-2]=table.cell_value(row,col)


for row in range(1,table.nrows):
    ys[row-1][0]=table.cell_value(row,1)
wid=table.ncols-2
# print(xs,ys,w)
N=500000000
w=np.ones((table.ncols-2,1))
# w=np.array([[89],[0.6]])
b=0.0
n=np.size(xs,0)

def cost(xs, ys, w):
    res=0.0
    for i in range(n):
        res+=(w.T.dot(xs[i].T)-ys[i])**2
    return math.sqrt(res/36.0)
a=2.5*1e-5
for i in range(N):
    tmp=cost(xs,ys,w)
    if(i%1000==0): print(tmp)

    grad=np.zeros((table.ncols-2,1))
    gb=0.
    for j in range(n):
        grad-=xs[j,:].reshape((wid,1))*(xs[j,:].reshape((1,wid)).dot(w)+b-ys[j].reshape((1,1)))
        gb-=xs[j,:].reshape((1,wid)).dot(w)+b-ys[j].reshape((1,1))
    w+=(a*grad)
    b+=a*gb
    if(tmp<cost(xs,ys,w)): break
print(w)
print(b)

# buybook.xlsx
# 25.311865257337942
# [[104.23255541]
#  [  0.40097719]]
# [[-0.00070219]]

# cement.xlsx
# 1.208888745433552
# [[2.19223092]
#  [1.15338833]
#  [0.7578144 ]
#  [0.48632662]]
# [[0.01095604]]