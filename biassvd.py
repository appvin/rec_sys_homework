import math
import random
import pandas as pd
from scipy.sparse import csr_matrix

def AddToDict(dic,i,j,v=-1):
    if v==-1:
        v=j
        if i not in dic:
            dic[i]=v
        else:
            dic[i]+=v
    else:
        if i not in dic:
            dic.update({i:{j: v}})
        elif j not in dic[i]:
            dic[i][j]=v
        else:
            dic[i][j]+=v

#u,i,rui,pui
def RMSE(records):
    rui=records[2]   
    pui=records[3]
    return math.sqrt(sum((rui-pui).dot(rui-pui))/float(len(records)))

def InitBiasLFM(train,F):
    p=dict()
    q=dict()
    bu=dict()
    bi=dict()
    #mu=mean(train[2])
    mu=3
    for index,row in train.iterrows():
        u=row[0]
        i=row[1]
        bu[u]=0
        bi[i]=0
        if u not in p:
            p[u]=[random.random()/math.sqrt(F) for x in range(0,F)]
        if i not in q:
            q[i]=[random.random()/math.sqrt(F) for x in range(0,F)]
    return [p,q,bu,bi,mu]

def Predict(u,i,p,q,bu,bi,mu):
    if u not in p:
        #print(['lost u',u])
        return mu
    if i not in q:
        #print(['lost i',i])
        return mu
    ret=mu+bu[u]+bi[i]
    ret+=sum(p[u][f]*q[i][f] for f in range(0,len(p[u])))
    return ret

def LearningBiasLFM(train,F,n,alpha,lamda):
    [p,q,bu,bi,mu]=InitBiasLFM(train,F)
    print('learning:')
    for step in range(0,n):
        print(step)
        for index,row in train.iterrows():
            u=row[0]
            i=row[1]
            rui=row[2]
            pui=Predict(u,i,p,q,bu,bi,mu)
            eui=rui-pui
            bu[u] += alpha * (eui - lamda * bu[u])
            bi[i] += alpha * (eui - lamda * bi[i]) 
            for k in range(0,F):
                p[u][k]+=alpha*(q[i][k]*eui-lamda*p[u][k])
                q[i][k]+=alpha*(p[u][k]*eui-lamda*q[i][k])
        alpha*=0.9
    return [bu,bi,mu,p,q]

def biassvd(train):
    [bu,bi,mu,p,q]=LearningBiasLFM(train,1,1,0.02,0.01)
    result=dict()
    nUsers = max(train[0])+1
    nItems = max(train[1])+1
    for u in range(1,nUsers):
        for i in range(1,nItems):
            if u not in result:
                result.update({u:{i: Predict(u,i,p,q,bu,bi,mu)}})
            else:
                result[u][i]=Predict(u,i,p,q,bu,bi,mu)
    return result
