#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:35:06 2018

@author: nam
"""
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

# This function will duplicate data for each value in y.
def duplicate_data(X_pt, y):
    X_new = []
    y_new = []
    duplicate_train_data = "/home/nam/Desktop/AspiringMinds/lstm_datafiles/duplicate_data/lstm_train_duplicate_data.txt"
    fp = open(duplicate_train_data,'w')
    duplicate_train_scores = "/home/nam/Desktop/AspiringMinds/lstm_datafiles/duplicate_data/lstm_train_duplicate_scores.txt"
    wp = open(duplicate_train_scores,'w')
    # Since we have 
    weights = [2,12,2,2,0,8,24] # Classes = [0,1,2,3,4,5,6]
    for data,score in zip(X_pt,y):
        if score<0.0:
            continue
        pickle.dump(data,fp)
        pickle.dump(score,wp)
        X_new.append(data)
        y_new.append(score)
        if weights[int(score)] == 0.0:
            continue
        for i in range(weights[int(score)]-1):
            pickle.dump(data,fp)
            pickle.dump(score,wp)
            X_new.append(data)
            y_new.append(score)
    fp.close()
    wp.close()
    return X_new, y_new
        
# This function takes mean of all essay lengths
def calculate_mean(X):
    total = len(X)
    length = 0
    for essay in X:
        length = length + len(essay)
    
    mean = length/total
    
    return mean

        
# Loading training data and scores - padding and truncating shorter and larger essays
def pad_and_truncate():
    fp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_train_data.txt','rb')
    X = []
    while True:
            try:
                X.append(pickle.load(fp))
            except EOFError:
                break
    wp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_train_scores.txt', 'rb')
    y = []
    while True:
            try:
                y.append(np.round(pickle.load(wp)))
            except EOFError:
                break
    fp.close()
    wp.close()
    
    return X, y
        

X, y = pad_and_truncate()

X_l, y_l = duplicate_data(X, y)

y_l =  [np.round(x) for x in y_l]
line = np.linspace(1,len(y_l), len(y_l))
plt.hist(y_l)
plt.show()
