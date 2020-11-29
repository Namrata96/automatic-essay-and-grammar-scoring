#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:05:21 2018

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
LEARNING_RATE = 1e-7
NUMBER_EPOCHS = 10
BATCH_SIZE = 5
ESSAY_SIZE = 300

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
    filep = open('/home/nam/Desktop/AspiringMinds/dataset/num_to_word.txt','w')
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
        


#data_preprocess()
#load_and_split_data()


X, y = pad_and_truncate()

model = lstm_model()

rmsprop = RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-7, decay=0.0)
#    sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.1,nesterov=True)
#adam = Adam(lr=3e-4)
train_checkpoint_file = '/home/nam/Desktop/AspiringMinds/lstm_datafiles/batch_lstm_weights/batch_train_weights.{epoch:02d}--{loss:.2f}.hdf5'
train_checkpoint = ModelCheckpoint(train_checkpoint_file, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
model.compile(optimizer = rmsprop, loss='mse', metrics=['accuracy'])
#    model.summary()
#    model.metrics_names
model.fit(X, y, epochs=NUMBER_EPOCHS, verbose=1, batch_size=BATCH_SIZE,callbacks=[train_checkpoint])
#    print "Set:" + str(old_weights)
    
# serialize model to JSON
model_json = model.to_json()
with open("/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_batch_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_batch_model.h5")
print("Saved model to disk")





