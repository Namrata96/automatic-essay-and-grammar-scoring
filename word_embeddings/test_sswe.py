#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:58:21 2017

@author: nam
"""

# Loading SSWE model
from keras import backend as K
from keras.models import model_from_json
from copy import deepcopy
import numpy as np
import pickle,csv
from keras.layers import Input, Embedding, Dense, Lambda, Flatten
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras import losses
def htanh(x):
    return K.clip(x, -1, 1)
ff = open("/home/nam/Desktop/AspiringMinds/dataset/sswe_model_all.json","r")
json_string = ff.read()
ff.close()
sswe_model = model_from_json(json_string, custom_objects={'htanh':htanh})
sswe_model.load_weights("/home/nam/Desktop/AspiringMinds/dataset/sswe_model_all.h5")

#
#def test_generator(batch_size):
#    fp1 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/test_true_sequences.txt', 'rb')
#    fp2 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/test_true_labels.txt', 'rb')
#    X = list()
#    y = list()
#    while True:
#        try:
#            X.append(pickle.load(fp1))
#        except EOFError:
#            break
#    while True:
#        try:
#            y.append(pickle.load(fp2))
##            print y[0]
#        except EOFError:
#            break
#    
#    length = len(X)
#    vocab_len = 25930 # dont change it to VOCAB_SIZE
#    while True:
#        batch_true = []
#        batch_noisy = []
#        batch_labels = []
#        useless_labels = []
#        for i in range(0,batch_size,200):
#            index = int(np.random.choice(length, 1))
#            for j in range(200):
#                batch_true.append(X[index])
#                batch_labels.append(y[index])
#                useless_labels.append(0)
#                noisy_word = int(np.random.choice(vocab_len, 1))
#                true_sequence = X[index]
#                noisy_sequence = deepcopy(true_sequence)
#                noisy_sequence[WINDOW_SIZE/2] = noisy_word
#                batch_noisy.append(noisy_sequence)
#        batch_true = np.asarray(batch_true)
#        batch_noisy = np.asarray(batch_noisy)    
#        yield [batch_true, batch_noisy], [np.asarray(useless_labels, dtype='float32'), np.asarray(batch_labels, dtype='float32')]
#
#
#def test_generator_single():
#    fp1 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/test_true_sequences.txt', 'rb')
#    fp2 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/test_true_labels.txt', 'rb')
#    X = list()
#    y = list()
#    X.append(pickle.load(fp1))
#    y.append(pickle.load(fp2))
##            print y[0]
##    length = len(X)
#    vocab_len = 25930 # dont change it to VOCAB_SIZE
#    while True:
#        print "Inside"
#        batch_true = []
#        batch_noisy = []
#        batch_labels = []
#        useless_labels = []
#        #    index = int(np.random.choice(length, 1))
#        batch_true.append(X[0])
#        batch_labels.append(y[0])
#        useless_labels.append(0)
#        noisy_word = int(np.random.choice(vocab_len, 1))
#        true_sequence = X[0]
#        noisy_sequence = deepcopy(true_sequence)
#        noisy_sequence[WINDOW_SIZE/2] = noisy_word
#        batch_noisy.append(noisy_sequence)
#        batch_true = np.asarray(batch_true)
#        batch_noisy = np.asarray(batch_noisy)    
##        print batch_true
##        print batch_noisy
#        yield [batch_true, batch_noisy], [np.asarray(useless_labels, dtype='float32'), np.asarray(batch_labels, dtype='float32')]

NEGATIVE_SAMPLES = 200
BATCH_SIZE = 1000
WINDOW_SIZE = 9
ALPHA = 0.1
#metrics = sswe_model.evaluate_generator(test_generator(TEST_SEQ_TOTAL//BATCH_SIZE+1), steps=1, verbose=1)
#y_pred = predict_generator(test_generator(TEST_SEQ_TOTAL//BATCH_SIZE+1), steps=1, verbose=1)
#print y_pred


filep = open('/home/nam/Desktop/AspiringMinds/dataset/num_to_word.txt','r')
num_to_word = pickle.load(filep)
filep.close()
def convert_to_words(m,n):
    m = m.tolist()
    n = n.tolist()
    a = list()
    b = list()
    m = m[0]
    n = n[0]
    for i, word_val in enumerate(m):
        if word_val == 0:
            a.append('pad')
        else:
            a.append(num_to_word[word_val])
    for i, word_val in enumerate(n):
        if word_val == 0:
            b.append('pad')
        else:
            b.append(num_to_word[word_val])
#    print a
    return a,b
    

sgd = Adam(lr=3e-4)
sswe_model.compile(optimizer=sgd, loss={'context_layer': lambda y_true, y_pred:y_pred,'output_layer': 'mean_squared_error'}, loss_weights=[ALPHA, 1-ALPHA], metrics={'context_layer': 'accuracy', 'output_layer': 'accuracy'})

fp1 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/test_true_sequences.txt', 'rb')
fp2 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/test_true_labels.txt', 'rb')
X = list()
y = list()
while True:
    try:
        X.append(pickle.load(fp1))
    except EOFError:
        break
while True:
    try:
        y.append(pickle.load(fp2))
#            print y[0]
    except EOFError:
        break

length = len(X)
print length
vocab_len = 25931
MAX_ITERATIONS = 3000
i=0
batch_true = []
batch_noisy = []
batch_labels = []
useless_labels = []
while i<MAX_ITERATIONS:
    batch_true.append(X[i])
    batch_labels.append(y[i])
    useless_labels.append(0)
    noisy_word = int(np.random.choice(vocab_len, 1))
    true_sequence = X[i]
    noisy_sequence = deepcopy(true_sequence)
    noisy_sequence[WINDOW_SIZE/2] = noisy_word
    batch_noisy.append(noisy_sequence)
    i = i + 1
batch_true = np.asarray(batch_true)
batch_noisy = np.asarray(batch_noisy)   
useless_labels = np.asarray(useless_labels,dtype='float32')
batch_labels = np.asarray(batch_labels, dtype='float32')

#    yield [batch_true, batch_noisy], [np.asarray(useless_labels, dtype='float32'), np.asarray(batch_labels, dtype='float32')]
wp = open('/home/nam/Desktop/AspiringMinds/dataset/test_results_all.csv', 'w')
csvwriter =csv.writer(wp, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
csvwriter.writerow(['true','noisy','context_loss','actual_score','pred_score','score_loss','loss'])

for batch_true_seq, batch_noisy_seq, useless_label, batch_label in zip(batch_true,batch_noisy,useless_labels,batch_labels):
    batch_true_seq = batch_true_seq.reshape(1,batch_true_seq.shape[0])
    batch_noisy_seq = batch_noisy_seq.reshape(1,batch_noisy_seq.shape[0])
    batch_label = batch_label.reshape(1)
    useless_label = useless_label.reshape(1)
    metrics = sswe_model.evaluate([batch_true_seq,batch_noisy_seq],
                                   [useless_label,batch_label])
#print sswe_model.metrics_names
#    print metrics
#    print batch_true_seq.shape
    y_pred = sswe_model.predict([batch_true_seq,batch_noisy_seq])
#    print y_pred
    m,n = convert_to_words(batch_true_seq,batch_noisy_seq)
    csvwriter.writerow([m,n,metrics[1],batch_label,y_pred[1],metrics[2],metrics[0]])
wp.close()