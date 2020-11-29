#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 11:01:01 2017

@author: nam
"""
import pickle
from gensim.models import Word2Vec
fp1 = open('/home/nam/Desktop/AspiringMinds/dataset/corpus_preprocessed.txt', 'r') 
corpus = []
while True:
    try:
        corpus.append(pickle.load(fp1))
    except EOFError:
        break
fp1.close()

word2vec_corpus = []
for essay in corpus:
    essay = essay.split()
    word2vec_corpus.append(essay)

# Training using skip gram model. for cbow change sg to 0.
model = Word2Vec(word2vec_corpus, size=200, min_count=1, window=4, sg=1, workers=3)

model.save('/home/nam/Desktop/AspiringMinds/dataset/word2vec_essay_model.bin')
## load model
#new_model = Word2Vec.load('model.bin')
    