#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:40:13 2018

@author: nam
"""

from keras import backend as K
from keras import losses
from keras.optimizers import RMSprop, SGD, Adam
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Bidirectional, LSTM,Dropout
from keras.callbacks import ModelCheckpoint
import pickle
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle,csv

VOCAB_SIZE = 44
DROPOUT_RATE =  0.5
EMBEDDING_SIZE = 45
LEARNING_RATE = 10e-7
NUMBER_EPOCHS = 1

filep = open('/home/nam/Desktop/AspiringMinds/dataset/num_to_word.txt','r')
num_to_word = pickle.load(filep)
filep.close()
def convert_to_words(m):
    m = m[0]
    a = list()
    for i, word_val in enumerate(m):
        if word_val == 0:
            a.append('pad')
        else:
            a.append(num_to_word[word_val])
   
#    print a
    return a


embedding_weights = list()
for i in range(VOCAB_SIZE+1):
    class_weights = list()
    for j in range(VOCAB_SIZE+1):
        if j>0 and j==i:
            class_weights.append(1)
        else:
            class_weights.append(0)
    embedding_weights.append(class_weights)
embedding_weights = np.asarray(embedding_weights)

total_sum_mse = 0
total_sum = 0
total_sum_abs = 0

def lstm_model(X,y,index):
    essay_size = X.shape[1]
    essay = Input(shape=(essay_size,), dtype='float32', name='essay')
    embedding_layer = Embedding(output_dim=EMBEDDING_SIZE, input_dim=VOCAB_SIZE+1, input_length=essay_size, weights=[embedding_weights], name='embedding_layer', trainable=False)
    essay_embedded = embedding_layer(essay)
    
    first_lstm_layer = Bidirectional(LSTM(10,return_sequences=True, name='first_lstm'), merge_mode='concat')
    temp_out_1 = first_lstm_layer(essay_embedded)
    dropout_layer = Dropout(DROPOUT_RATE,name='first_dropout_layer')
    first_lstm_out = dropout_layer(temp_out_1)
    
    second_lstm_layer = Bidirectional(LSTM(10, name='first_lstm'), merge_mode='concat')
    temp_out_2 = second_lstm_layer(first_lstm_out)
    dropout_layer = Dropout(DROPOUT_RATE,name='second_dropout_layer')
    second_lstm_out = dropout_layer(temp_out_2)
    dense_layer = Dense(1, name='output_layer')
    out = dense_layer(second_lstm_out)

    model = Model(inputs=essay, outputs=out)
    
    rmsprop = RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-7, decay=0.0)
#    sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.1,nesterov=True)
#    adam = Adam(lr=3e-4)
#    old_weights = open("/home/nam/Desktop/AspiringMinds/dataset/sswe_model_all.h5",'r')
    
    model.compile(optimizer = rmsprop, loss='mse', metrics=['accuracy'])
    model.load_weights("/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_weights/train_weights_3039_.00--1.49.hdf5")
#    model.summary()
#    model.metrics_names
    metrics = model.evaluate(X,y)
    y_pred = model.predict(X)
    wp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_results/test_results.csv', 'a')
    csvwriter = csv.writer(wp, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    if index==0:
        csvwriter.writerow(['essay','actual_score','predicted_score','loss'])
    m = convert_to_words(X)
    csvwriter.writerow([m,y,y_pred,metrics[0]])
    global total_sum_mse
    total_sum_mse = total_sum_mse + metrics[0]
    global total_sum
    total_sum = total_sum + (y-y_pred)
    global total_sum_abs
    total_sum_abs = total_sum_abs + abs(y-y_pred)
    wp.close()
    
# Loading training data and scores
fp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_test_data.txt','rb')
X = []
while True:
        try:
            X.append(pickle.load(fp))
        except EOFError:
            break
wp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_test_scores.txt', 'rb')
y = []
while True:
        try:
            y.append(pickle.load(wp))
        except EOFError:
            break
fp.close()
wp.close()


for idx,val in enumerate(X):
    X[idx] = np.array(val)
    X[idx] = X[idx].reshape(1,X[idx].shape[0])
    y[idx] = np.array(y[idx])
    y[idx] = y[idx].reshape(1)
    lstm_model(X[idx],y[idx],idx)