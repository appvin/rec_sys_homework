import math
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging
from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

class item_search:
    index=0
    dictionary=0
    lsi=0
    
    def __init__(self,itempath):
        item_search.items = pd.read_table(itempath, header=None,sep='|')
        item_topic=item_search.items[1].str.partition('(')[0]

        #小写化，去停用词，去标点符号
        texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in item_topic]  
        english_stopwords = stopwords.words('english')
        texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_filtered_stopwords]

        #词干化
        item_search.st = LancasterStemmer()
        texts_stemmed = [[item_search.st.stem(word) for word in docment] for docment in texts_filtered]

        #去掉语料库中的低频词
        '''all_stems = sum(texts_stemmed)
        stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
        texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]'''
        texts=texts_stemmed

        #gensim预处理
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        item_search.dictionary = corpora.Dictionary(texts)
        corpus = [item_search.dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        #训练topic为10的LSI模型，建立索引
        item_search.lsi = models.LsiModel(corpus_tfidf, id2word=item_search.dictionary, num_topics=10) 
        item_search.index = similarities.MatrixSimilarity(item_search.lsi[corpus],num_features=10)
    
    def search(self,srchstr,K=10):
        ml_item=srchstr.partition('(')[0]
        #print(ml_item)
        texts_tokenized = [word.lower() for word in word_tokenize(ml_item)]  
        english_stopwords = stopwords.words('english')
        texts_filtered_stopwords = [word for word in texts_tokenized if not word in english_stopwords]
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        texts_filtered = [word for word in texts_filtered_stopwords if not word in english_punctuations]

        #词干化
        ml_item = [item_search.st.stem(word) for word in texts_filtered]
        #查询前K个最相似的
        #print(ml_item)
        ml_bow = item_search.dictionary.doc2bow(ml_item)
        ml_lsi = item_search.lsi[ml_bow]
        sims = item_search.index[ml_lsi]
        #print(sims)
        sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
        #print(sort_sims)
        top=[]
        idx=0
        for i in sort_sims:
            top.append(item_search.items.loc[i[0]][1])
            idx+=1
            if idx==K:
                break
        return top
