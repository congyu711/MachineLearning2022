import boost_pythonSVM
import numpy as np
import os
from skimage import feature as ft
from skimage import color
from skimage import io

fds=[]
ys=[]
with open("./data/train/train.txt") as trainfiles:
    for line in trainfiles:
        strs=line.split(sep=' ')
        # print(strs)
        img=io.imread("./data/train/"+strs[0]+".jpg")
        gray=color.rgb2gray(img)/255.0
        fd=ft.hog(gray, orientations=12, block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4,4], visualize=False, transform_sqrt=True)
        fd=np.array(fd)
        fds.append(fd)
        ys.append(strs[1])

f = open("train.data", "w")
for i in range(len(fds)):
    for e in fds[i]:
        f.write(str(e)+' ')
    f.write(ys[i])
svm=boost_pythonSVM.svm(102,22464,"train.data")
svm.train(1000)
f.close()

ffds=[]
yys=[]
with open("./data/test/test.txt") as testfile:
        for line in testfile:
            strs=line.split(sep=' ')
            img=io.imread("./data/test/"+strs[0]+".jpg")
            gray=color.rgb2gray(img)/255.0
            fd=ft.hog(gray, orientations=12, block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4,4], visualize=False, transform_sqrt=True)
            fd=np.array(fd)
            ffds.append(fd)
            yys.append(strs[1])

f = open("test.data", "w")
for i in range(len(ffds)):
    for e in ffds[i]:
        f.write(str(e)+' ')
    f.write(yys[i])
f.close()

svm.test("test.data",14)