#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:47:46 2018

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

VOCAB_SIZE = 44
DROPOUT_RATE =  0.5
EMBEDDING_SIZE = 45
LEARNING_RATE = 10e-7
NUMBER_EPOCHS = 1

# This function loads the POS tagged corpus and splits the data into train and test sets
def load_and_split_data():
    # Loading dtagged ata and scores
    fp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_tagged_corpus.txt','r')
    wp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/grammar_scores.txt', 'r')
    lines = list()
    while True:
        try:
            lines.append(pickle.load(fp))
        except EOFError:
            break
    score_lines = list()
    while True:
        try:
            score_lines.append(pickle.load(wp))
        except EOFError:
            break
    number_essays = len(lines)
    fp.close()
    fp = None
    
    # Finding all different types of tags to integer code them.
    vocab = set()
    for essay in lines:
        essay = essay.split()
        for tag in essay:
            vocab.add(tag)
    
    # Integer coding each pos tag.
    tag_to_integer = dict()
    for i,tag in enumerate(vocab):
        tag_to_integer[tag] = i+1
    
    # Storing the reverse dictionary.
    num_to_tag = dict()
    for key in tag_to_integer.iterkeys():
        num_to_tag[tag_to_integer[key]] = key 
    filep = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/num_to_word.txt','w')
    pickle.dump(num_to_tag,filep)
    filep.close()
    
    integer_corpus = list()
    # Ineger coding the entire essay.
    for line in lines:
        line = line.split()
        new_essay = list()
        for tag in line:
            new_essay.append(tag_to_integer[tag])
        integer_corpus.append(new_essay)
    
    #Train test split
    train_data = list()
    test_data = list()
    train_percentage = 0.7
    max_count = int(train_percentage*number_essays)
    train_data = integer_corpus[:max_count]
    test_data = integer_corpus[max_count:]
    train_scores = score_lines[:max_count]
    test_scores = score_lines[max_count:]
    
    # Storing test and train data
    fp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_train_data.txt','wb')
    wp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_test_data.txt','wb')
    rp1 = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_train_scores.txt','wb')
    rp2 = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_test_scores.txt','wb')
    for data in train_data:
        pickle.dump(data,fp)
    for data in test_data:
        pickle.dump(data,wp)
    for data in train_scores:
        pickle.dump(data,rp1)
    for data in test_scores:
        pickle.dump(data,rp2)
    fp.close()
    wp.close()
    rp1.close()
    rp2.close()
    
    
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

old_weights = None
# Creating the model
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
    train_checkpoint_file = '/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_weights/train_weights.{epoch:02d}--{loss:.2f}.hdf5'
    train_checkpoint = ModelCheckpoint(train_checkpoint_file, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
    if index>0:
#         print "assign: " + str(old_weights)
         model.set_weights(old_weights)
    model.compile(optimizer = rmsprop, loss='mse', metrics=['accuracy'])
#    model.summary()
#    model.metrics_names
    
    model.fit(X, y, epochs=1, verbose=1, batch_size=1,callbacks=[train_checkpoint])
    model.reset_states()
    global old_weights
    old_weights = model.get_weights()
#    print "Set:" + str(old_weights)
   

#load_and_split_data()

# Loading training data and scores
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
            y.append(pickle.load(wp))
        except EOFError:
            break
fp.close()
wp.close()
# Set VOCAB_SIZE
determine_vocab_size(X)

# Creating model

#model = lstm_model()


#X = [[1,2,3,4,5],[2,3,4]]
#y = [1.,2.]
for idx,val in enumerate(X):
    X[idx] = np.array(val)
    X[idx] = X[idx].reshape(1,X[idx].shape[0])
    y[idx] = np.array(y[idx])
    y[idx] = y[idx].reshape(1)
    lstm_model(X[idx],y[idx],idx)
X = np.asarray(X)
X = X.reshape(1,X.shape[0])
y = np.asarray(y) 
y = y.reshape(1,y.shape[0])


    
