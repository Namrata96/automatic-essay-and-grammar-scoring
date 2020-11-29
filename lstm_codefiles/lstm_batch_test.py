#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:49:26 2018

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
from keras.preprocessing.sequence import pad_sequences
import numpy as np

VOCAB_SIZE = 44
DROPOUT_RATE =  0.5
EMBEDDING_SIZE = 45
LEARNING_RATE = 1e-2
NUMBER_EPOCHS = 3000
BATCH_SIZE = 10
ESSAY_SIZE = 300

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
    mean = calculate_mean(X) # use this to calculate the essay size 
    global ESSAY_SIZE
    ESSAY_SIZE = mean
    X_pt = []
    X_pt = pad_sequences(X,maxlen=ESSAY_SIZE,truncating='post',dtype='float64')
    return X_pt, np.asarray(y,dtype='float64')
        

# Creating the model
def lstm_model():
    essay = Input(shape=(ESSAY_SIZE,), dtype='float32', name='essay')
    embedding_layer = Embedding(output_dim=EMBEDDING_SIZE, input_dim=VOCAB_SIZE+1, input_length=ESSAY_SIZE, weights=[embedding_weights], name='embedding_layer', trainable=False)
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
    return model
    


X, y = pad_and_truncate()

model = lstm_model()

rmsprop = RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-7, decay=0.0)
#    sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.1,nesterov=True)
#adam = Adam(lr=3e-4)
model.load_weights("/home/nam/Desktop/AspiringMinds/lstm_datafiles/duplicate_data/lstm_duplicate_classify_batch_model.h5")
model.compile(optimizer = rmsprop, loss='mse', metrics=['accuracy'])
wp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/duplicate_data/test_results_batch.csv', 'w')
csvwriter = csv.writer(wp, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
csvwriter.writerow(['essay','actual_score','predicted_score','loss'])
total_sum_mse = 0
total_sum = 0
total_sum_abs = 0
for x, score in zip(X,y):
    x = np.array(x)
    x = x.reshape(1, x.shape[0])
    score = np.array(score)
    score = score.reshape(1)
    metrics = model.evaluate(x,score)
    y_pred = model.predict(x)
    m = convert_to_words(x)
    total_sum_mse = total_sum_mse + metrics[0]
    total_sum = total_sum + (score-y_pred)
    total_sum_abs = total_sum_abs + abs(score-y_pred)
    csvwriter.writerow([m,score,y_pred,metrics[0]])
csvwriter.writerow(["MSE loss:", total_sum_mse])
csvwriter.writerow(["Difference", total_sum])
csvwriter.writerow(["AbsoluteDifference", total_sum_abs])
wp.close()
    
    