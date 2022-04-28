import os
os.environ["PATH"] += os.pathsep + 'C:/Users/15840/AppData/Local/Programs/Python/Python310/lib/site-packages/graphviz/'

import math
from random import shuffle
import numpy as np
np.set_printoptions(linewidth=400)
from graphviz import Digraph

class node:
    def __init__(self) -> None:
        self.isleaf = 0
        self.sonlist = []
        self.sonatt = []
        self.no=-1
        self.decatt = -1
        self.y = -1

    def _pirnt(self):
        print("----")
        print("No",self.no)
        print("isleaf", self.isleaf)
        print("dec_attribute", self.decatt)
        print("sonatt", self.sonatt)
        print("sonlist", self.sonlist)
        print("y", self.y)
        print("----")


attributes = {'age': 0, 'job': 1, 'house': 2, 'debet': 3, 'y': 4}

values = [['qn', 'zn', 'ln'], ['0', '1'], ['0', '1'],
          ['normal', 'good', 'perfect'], ['0', '1']]
U = []

with open('load_data.csv') as file:
    strs = file.readlines()
    for i in range(len(strs)):
        strs[i] = strs[i].strip('\n')
    attributes = strs[0].split(',')[1:]
    # print(attributes)
    U = [['' for i in range(5)] for i in range(len(strs))]
    for i in range(1, len(strs)):
        tmp = strs[i].split(',')
        for j in range(5):
            U[i][j] = tmp[j+1]


pos = []
neg = []
for i in range(1, 16):
    if(U[i][4] == '1'):
        pos.append(U[i])
    else:
        neg.append(U[i])

shuffle(neg)
shuffle(pos)
# print(neg)
p = 0.3
train = []
test = []

for i in range(len(neg)):
    if(i <= len(neg)*p):
        test.append(neg[i])
    else:
        train.append(neg[i])
for i in range(len(pos)):
    if(i <= len(pos)*p):
        test.append(pos[i])
    else:
        train.append(pos[i])

root = node()

idx=0
def treegen(h: node, D, A: set) -> node:
    global idx
    h.no=idx
    idx=idx+1

    f0 = 0
    f1 = 0
    for i in D:
        if(i[4] != '0'):
            f0 = 1
        if(i[4] != '1'):
            f1 = 1
    if(not f0):
        h.y = 0
        h.isleaf = 1
        return h
    if(not f1):
        h.y = 1
        h.isleaf = 1
        return h
    f = 0
    for i in range(1, len(D)):
        if(D[i][4] != D[0][4]):
            f = 1
            break
    if(len(A) == 0 or not f):
        h.isleaf = 1
        cnt = [0, 0]
        for i in D:
            if(i[4] == '0'):
                cnt[0] = cnt[0]+1
            else:
                cnt[1] = cnt[1]+1
        if(cnt[0] > cnt[1]):
            h.y = 0
        else:
            h.y = 1
        return h
    # ent&gain
    ents = {}  # find min
    for att in A:
        num = {x: [0, 0] for x in values[att]}

        for obj in D:  # get |Dv|
            if(obj[4] == '1'):
                num[obj[att]][1] = num[obj[att]][1]+1
            else:
                num[obj[att]][0] = num[obj[att]][0]+1
        # print(att,num)
        ent_a = 0.0
        for r, l in num.items():  # Dv
            ent_l = 0.0
            s = sum(l)
            if(s == 0):
                continue
            for ll in l:
                if(ll/s == 0):
                    continue
                ent_l -= (ll/s)*math.log(ll/s)
            ent_a += ent_l
        ents[att] = ent_a
    index = min(ents, key=ents.get)
    # print(ents)
    # print(index)
    h.decatt = index
    subD = {x: [] for x in values[index]}
    for obj in D:
        subD[obj[index]].append(obj)
    # print(index)
    # print("---\n subD",subD)
    A.remove(index)
    for att, subset in subD.items():
        if(len(subset) == 0):
            res = node()
            res.no=idx
            idx=idx+1
            res.isleaf = 1
            cnt = [0, 0]
            for i in D:
                if(i[4] == '0'):
                    cnt[0] = cnt[0]+1
                else:
                    cnt[1] = cnt[1]+1
            if(cnt[0] > cnt[1]):
                res.y = 0
            else:
                res.y = 1
            h.sonlist.append(res)
            h.sonatt.append(att)
        else:
            res = node()
            h.sonlist.append(treegen(res, subset, A))
            h.sonatt.append(att)
    return h


# for i in train: print(i)
attset = {0, 1, 2, 3}
treegen(root, train, attset)



dot=Digraph()
def dfs(h: node):
    # h._pirnt()
    dot.node(str(h.no),str(h.decatt))
    for i in range(len(h.sonlist)):
        dfs(h.sonlist[i])
        dot.edge(str(h.no),str(h.sonlist[i].no),str(h.sonatt[i]))

dfs(root)
print(dot.source)
dot.view()

for i in range(1, len(U)):
    p=root
    while(1):
        if(p.isleaf):
            print(U[i][4], p.y)
            break
        for j in range(len(p.sonatt)):
            if(p.sonatt[j]==U[i][p.decatt]):
                p=p.sonlist[j]
                break
        