import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class itemsim:
    S=0
    
    def __init__(self,itempath):
        itemsim.items = pd.read_table(itempath, header=None,sep='|') 
        item_genre=itemsim.items.loc[:,5:23]
        itemsim.S=cosine_similarity(item_genre)
    
    def topn(self,item,K=10):
        item-=1
        l=pd.DataFrame(itemsim.S[item])      
        l[1]=l.index+1
        l[2]=itemsim.items[1]
        l=pd.DataFrame(l.sort_values(by=0,ascending=False))
        top=[]
        idx=0
        for index,row in l.iterrows():
            if row[1]==item+1:
                continue
            else:
                top.append(row[2])
            idx+=1
            if idx==K:
                break
        return top
