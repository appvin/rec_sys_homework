import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import distance
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

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

def h_user(uirt):
    uirt=uirt.copy()
    col=uirt.columns
    uirt.columns=['u','i','r','timestamp']
    all_u=uirt['u'].drop_duplicates()
    meanr=dict()
    stdr=dict()
    print('mean,std...')
    for row in all_u:
        u_r=uirt[uirt['u']==row]['r']
        meanr[row]=u_r.mean()
        stdr[row]=u_r.std()
    print('h...')
    for index,row in uirt.iterrows():
        if index%10000==0:
            print(index)
        uirt.loc[index,'r']=(row['r']-meanr[row['u']])/stdr[row['u']]
    uirt.columns=col
    return [uirt,meanr,stdr]

def _h_user(uirt,meanr,stdr):
    uirt=uirt.copy()
    col=uirt.columns
    uirt.columns=['u','i','r','e']
    all_u=uirt['u'].drop_duplicates()
    print('_h...')
    for index,row in uirt.iterrows():
        if index%10000==0:
            print(index)
        uirt.loc[index,'e']=row['e']*stdr[row['u']]+meanr[row['u']]
    uirt.columns=col
    return uirt

def h_item(uirt):
    uirt=uirt.copy()
    col=uirt.columns
    uirt.columns=['u','i','r','timestamp']
    all_i=uirt['i'].drop_duplicates()
    meanr=dict()
    stdr=dict()
    print('mean,std...')
    for row in all_i:
        u_r=uirt[uirt['i']==row]['r']
        meanr[row]=u_r.mean()
        stdr[row]=u_r.std()
    print('h...')
    for index,row in uirt.iterrows():
        if index%10000==0:
            print(index)        
        eui=(row['r']-meanr[row['i']])/stdr[row['i']]
        if np.isnan(stdr[row['i']]):
            eui=uirt.loc[index,'r']
        elif stdr[row['i']]==0:
            eui=uirt.loc[index,'r']
        uirt.loc[index,'r']=eui
    uirt.columns=col
    return [uirt,meanr,stdr]

def _h_item(uirt,meanr,stdr):
    uirt=uirt.copy()
    col=uirt.columns
    uirt.columns=['u','i','r','e']
    all_u=uirt['i'].drop_duplicates()
    print('_h...')
    for index,row in uirt.iterrows():
        if index%10000==0:
            print(index)
        if row['i'] not in stdr:
            continue
        uirt.loc[index,'e']
        eui=row['e']*stdr[row['i']]+meanr[row['i']]
        if np.isnan(stdr[row['i']]):
            eui=uirt.loc[index,'r']
            #print(['nan:',eui])
        elif stdr[row['i']]==0:
            eui=uirt.loc[index,'r']
            #print(['0:',eui])
        uirt.loc[index,'e']=eui
    uirt.columns=col
    return uirt
