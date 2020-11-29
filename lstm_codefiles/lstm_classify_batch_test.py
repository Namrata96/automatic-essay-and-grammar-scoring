#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:57:51 2018

@author: nam
"""
from keras import backend as K
from keras import losses
from keras.optimizers import RMSprop, SGD, Adam
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Bidirectional, LSTM,Dropout
import pickle, csv
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix

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

# This function determines the number of unique tokens in the dataset.
def determine_vocab_size(corpus):
    vocab = set()
    for essay in corpus:
        essay = essay.split()
        for word in essay:
            vocab.add(word)
    global VOCAB_SIZE
    VOCAB_SIZE = len(vocab)

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
    dense_layer = Dense(7, name='output_layer',activation='softmax')
    out = dense_layer(second_lstm_out)

    model = Model(inputs=essay, outputs=out)
    return model

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
                y.append(np.round(pickle.load(wp)))
            except EOFError:
                break
    fp.close()
    wp.close()
    mean = calculate_mean(X) # use this to calculate the essay size 
    global ESSAY_SIZE
    ESSAY_SIZE = mean
    X_pt = []
    X_pt = pad_sequences(X,maxlen=292,truncating='post',dtype='float64')
    return X_pt, np.asarray(y,dtype='float64')
        
X_p, y = pad_and_truncate()

model = lstm_model()

rmsprop = RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-7, decay=0.0)
model.load_weights("/home/nam/Desktop/AspiringMinds/lstm_datafiles/duplicate_data/duplicate_class_batch_train_weights_36.hdf5")
model.compile(optimizer = rmsprop, loss='mse', metrics=['accuracy'])
wp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/duplicate_data/test_essays_results_classify_batch_more_dup.csv', 'w')
csvwriter = csv.writer(wp, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
csvwriter.writerow(['essay'])
fp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/duplicate_data/test_metrics_results_classify_batch_more_dup.csv', 'w')
csvwriter2 = csv.writer(fp, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
csvwriter2.writerow(['actual_score','predicted_score','loss'])
total_sum_mse = 0
total_sum = 0
total_sum_abs = 0


# Evaluating on entire test set
y_pred = model.predict(X_p)
metrics = model.evaluate(X_p,to_categorical(y,num_classes=7))
print model.metrics_names
print metrics
cm = confusion_matrix(y , y_pred.argmax(axis=1))
print cm

# For seeing the distribution of essays in each class
#import pandas as pd
#df=pd.Series(y)
#print df.value_counts()




# Evaluating each sample and storing the results
for x, score in zip(X_p,y):
    print x
    print score
    if score<0.0:
        continue
    x = np.array(x)
    x = x.reshape(1, x.shape[0])
    score = np.array(score)
    score = score.reshape(1)
    metrics = model.evaluate(x,to_categorical(score, num_classes=7))
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred,axis=1)
    m = convert_to_words(x)
    total_sum_mse = total_sum_mse + metrics[0]
    total_sum = total_sum + (score-y_pred)
    total_sum_abs = total_sum_abs + abs(score-y_pred)
    csvwriter.writerow([m])
    csvwriter2.writerow([score,y_pred,metrics[0]])
csvwriter2.writerow(["MSE loss:", total_sum_mse])
csvwriter2.writerow(["Difference", total_sum])
csvwriter2.writerow(["AbsoluteDifference", total_sum_abs])
wp.close()
fp.close()

