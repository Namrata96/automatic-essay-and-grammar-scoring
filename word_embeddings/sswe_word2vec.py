#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:06:23 2017

@author: nam
"""

from keras import backend as K
from keras.preprocessing.text import Tokenizer, one_hot
import os, csv, re
from pandas import read_csv
from keras.preprocessing.sequence import pad_sequences
from copy import deepcopy
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.optimizers import SGD, Adam
from keras import losses
import pickle
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Lambda, Flatten
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

  
# Global variables
VOCAB_SIZE = 25931 #25930. +1 for start_end_token
WINDOW_SIZE = 9 
EMBEDDING_SIZE = 200
EMBEDDING_MATRIX_SIZE = EMBEDDING_SIZE * VOCAB_SIZE
ALPHA = 0.1
NEGATIVE_SAMPLES = 200

LEARNING_RATE = 3e-4 # This is not given in the paper, please suggest
BATCH_SIZE = 50000
INPUT_SIZE = WINDOW_SIZE * EMBEDDING_SIZE
INPUT_LAYER_NEURONS = 200
HIDDEN_LAYER_NEURONS = 100 
OUTPUT_SIZE = 1
TOKEN_WORD = 'pad'

NUMBER_EPOCHS  = 1

train_file = '/home/nam/Desktop/AspiringMinds/dataset/EssayGrading/training_set.tsv'
ids_file = '/home/nam/Desktop/AspiringMinds/dataset/EssayGrading/kaggle_ids.csv'
START_END_TOKEN = 0    
num_to_word = dict()
def data_preprocess():
    # Remove "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT"
    # Opening the training dataset
    print "Data preprocessing start."
    pandas_data = read_csv(train_file, delimiter='\t')
    essays = pandas_data['essay']
    final_scores_d1 = pandas_data['domain1_score']
    id_to_score = dict()
    final_scores_d2 = pandas_data['domain2_score']
    essay_ids = pandas_data['essay_id']
    for essay_id, score in zip(essay_ids, final_scores_d1):
        id_to_score[essay_id] = score
    train_set = set()
    test_set = set()
    validation_set = set()
    classify_data = read_csv(ids_file, delimiter=',')
    set_ids = classify_data['Set']
    ids = classify_data['ID'] 
    for set_id, ID in zip(set_ids, ids):
        if set_id == 'train':
            train_set.add(ID)
        elif set_id == 'test':
            test_set.add(ID)
        else:
            validation_set.add(ID)
    
    corpus = []
    # Counting total number of distinct words (including stop words)
    for essay, essay_id in zip(essays, essay_ids):
        # Removing @["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT"]
        essay = re.sub('@[A-Z]+[0-9]?',' ',essay)
        # Removing special characters etc
        essay = re.sub('[^a-zA-Z]',' ',essay)
        # converting everything to lower case
        essay = essay.lower()
        essay = essay.split() #splits the string into list of words separated by space
#        print str(essay)
        # Stemming 
        ps = PorterStemmer()
        essay = [ps.stem(word) for word in essay]
        essay = ' '.join(essay)
        corpus.append(essay)
    tokeniser.fit_on_texts(corpus)
    # To find out the last encoding number to determine the vocabulary size.
    VOCAB_SIZE = len(tokeniser.word_index) + 1
#    print tokeniser.word_index
    for key in tokeniser.word_index.iterkeys():
        num_to_word[tokeniser.word_index[key]] = key 
#    print num_to_word
    corpus = tokeniser.texts_to_sequences(corpus)
    for essay, essay_id in zip(corpus, essay_ids):
        if essay_id in train_set:
            train_corpus.append(essay)
            train_labels.append(id_to_score[essay_id])
        elif essay_id in test_set:
            test_corpus.append(essay)
            test_labels.append(id_to_score[essay_id])
        else:
            valid_corpus.append(essay)
            valid_labels.append(id_to_score[essay_id])
#    print str(corpus[-1])
    # This splits the sentences into words and lower cases them.
    # It removes special characters as well.
    print "Size of training data: " + str(len(train_corpus))
    print "Size of valdation data: " + str(len(valid_corpus))
    print "Size of test data: " + str(len(test_corpus))


def get_true_sequences():
    data_preprocess()
    print "Data Preprocessing done."
    print "Generating sequences."
    # Converting padding word to a number. 
    START_END_TOKEN = tokeniser.texts_to_sequences([TOKEN_WORD])
    print START_END_TOKEN
    # Training phase
    # For storing true sequences
    true_sequences = list()
    labels = list()
    # Generating true sequences
    for essay, train_label in zip(train_corpus, train_labels):
        length = len(essay)
        sequence = list()
        for i in range(WINDOW_SIZE/2):
            essay.insert(length+i,START_END_TOKEN[0][0])
        for i in range(WINDOW_SIZE/2):
            essay.insert(i,START_END_TOKEN[0][0])
        
        # This loop handles all sequences which require no padding
        for i in range(WINDOW_SIZE/2, WINDOW_SIZE/2+length):
            for j in range(i-WINDOW_SIZE/2, i):
                sequence.append(essay[j])
                
            for j in range(i, i+WINDOW_SIZE/2+1):
#                print j
                sequence.append(essay[j])
            true_sequences.append(sequence)
            labels.append(train_label)
            sequence = list()
    TRAIN_SEQ_TOTAL = len(true_sequences)  
#        print "Essay number: " + str(idx) 
    # Storing the train data
    fp1 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/train_true_sequences.txt', 'wb') 
    for sequence in true_sequences:
            pickle.dump(sequence, fp1)
    fp2 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/train_true_labels.txt', 'wb') 
    for label in labels:
            pickle.dump(label, fp2)
    fp1.close()
    fp2.close()
               
    # Validation phase
    # For storing true sequences
    true_sequences = list()
    labels = list()
    # Generating true sequences
    for essay, valid_label in zip(valid_corpus, valid_labels):
        length = len(essay)
        sequence = list()
        for i in range(WINDOW_SIZE/2):
            essay.insert(length+i,START_END_TOKEN[0][0])
        for i in range(WINDOW_SIZE/2):
            essay.insert(i,START_END_TOKEN[0][0])
        
        # This loop handles all sequences which require no padding
        for i in range(WINDOW_SIZE/2, WINDOW_SIZE/2+length):
            for j in range(i-WINDOW_SIZE/2, i):
                sequence.append(essay[j])
            for j in range(i, i+WINDOW_SIZE/2+1):
#                print j
                sequence.append(essay[j])
            true_sequences.append(sequence)
            labels.append(valid_label)
            sequence = list()
    VALID_SEQ_TOTAL = len(true_sequences) 
    # Storing the validation sequences
    fp1 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/valid_true_sequences.txt', 'wb') 
    for sequence in true_sequences:
            pickle.dump(sequence, fp1)
    fp2 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/valid_true_labels.txt', 'wb') 
    for label in labels:
            pickle.dump(label, fp2)
    fp1.close()
    fp2.close()
        
    # Testing phase
    # For storing true sequences
    true_sequences = list()
    labels = list()
    # Generating true sequences
    for essay, test_label in zip(test_corpus, test_labels):
        length = len(essay)
        sequence = list()
        for i in range(WINDOW_SIZE/2):
            essay.insert(length+i,START_END_TOKEN[0][0])
        for i in range(WINDOW_SIZE/2):
            essay.insert(i,START_END_TOKEN[0][0])
        
        # This loop handles all sequences which require no padding
        for i in range(WINDOW_SIZE/2, WINDOW_SIZE/2+length):
            for j in range(i-WINDOW_SIZE/2, i):
                sequence.append(essay[j])
            for j in range(i, i+WINDOW_SIZE/2+1):
#                print j
                sequence.append(essay[j])
            true_sequences.append(sequence)
            labels.append(test_label)
            sequence = list()
    TEST_SEQ_TOTAL = len(true_sequences)
    # Storing the test sequences
    fp1 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/test_true_sequences.txt', 'wb') 
    for sequence in true_sequences:
            pickle.dump(sequence, fp1)
    fp2 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/test_true_labels.txt', 'wb') 
    for label in labels:
            pickle.dump(label, fp2)
    fp1.close()
    fp2.close()
    print "True sequences generated and saved."
    return TRAIN_SEQ_TOTAL, VALID_SEQ_TOTAL, TEST_SEQ_TOTAL
            

def sparsify(y):
    # Returns labels in binary NumPy array
    n_classes = 61
    length = len(y)
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(length)])

def train_generator(batch_size):
    fp1 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/train_true_sequences.txt', 'rb')
    fp2 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/train_true_labels.txt', 'rb')
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
    vocab_len = len(tokeniser.word_index) # dont change it to VOCAB_SIZE
    while True:
        batch_true = []
        batch_noisy = []
        batch_labels = []
        useless_labels = []
        for i in range(0,batch_size,200):
            index = int(np.random.choice(length, 1))
            for j in range(200):
                batch_true.append(X[index])
                batch_labels.append(y[index])
                useless_labels.append(0)
                noisy_word = int(np.random.choice(vocab_len, 1))
                true_sequence = X[index]
                noisy_sequence = deepcopy(true_sequence)
                noisy_sequence[WINDOW_SIZE/2] = noisy_word
                batch_noisy.append(noisy_sequence)
#        print "hi"
        batch_true = np.asarray(batch_true)
        batch_noisy = np.asarray(batch_noisy) 
        yield [batch_true, batch_noisy], [np.asarray(useless_labels, dtype='float32'), np.asarray(batch_labels, dtype='float32')]
    
def valid_generator(batch_size):
    fp1 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/valid_true_sequences.txt', 'rb')
    fp2 = open('/home/nam/Desktop/AspiringMinds/word_embeddings/valid_true_labels.txt', 'rb')
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
    vocab_len = len(tokeniser.word_index) # dont change it to VOCAB_SIZE
    while True:
        batch_true = []
        batch_noisy = []
        batch_labels = []
        useless_labels = []
        for i in range(0,batch_size,200):
            index = int(np.random.choice(length, 1))
            for j in range(200):
                batch_true.append(X[index])
                batch_labels.append(y[index])
                useless_labels.append(0)
                noisy_word = int(np.random.choice(vocab_len, 1))
                true_sequence = X[index]
                noisy_sequence = deepcopy(true_sequence)
                noisy_sequence[WINDOW_SIZE/2] = noisy_word
                batch_noisy.append(noisy_sequence)
        batch_true = np.asarray(batch_true)
        batch_noisy = np.asarray(batch_noisy)    
        yield [batch_true, batch_noisy], [np.asarray(useless_labels, dtype='float32'), np.asarray(batch_labels, dtype='float32')]
            
    
def customized_loss(args):
    y_true, y_pred = args
    loss = K.mean(K.maximum(1. - y_true + y_pred, 0.), axis=-1)
    print loss
    return loss

def htanh(x):
    return K.clip(x, -1, 1)

def CandW_model():
    true_sequence = Input(shape=(WINDOW_SIZE,), dtype='int32', name='true_sequence')
    noisy_sequence = Input(shape=(WINDOW_SIZE,), dtype='int32', name='noisy_sequence')
    
    shared_embedding = Embedding(output_dim=EMBEDDING_SIZE, input_dim=VOCAB_SIZE, input_length=WINDOW_SIZE, name='embedding_layer')
    encode_true = shared_embedding(true_sequence)
    encode_noisy = shared_embedding(noisy_sequence)
    
    shared_flatten = Flatten(name='flatten_layer')
    true_flattened = shared_flatten(encode_true)
    noisy_flattened = shared_flatten(encode_noisy)
    
    shared_input_layer = Dense(INPUT_LAYER_NEURONS, activation= htanh, name='input_layer')
    true_input = shared_input_layer(true_flattened)
    noisy_input= shared_input_layer(noisy_flattened)
    
    shared_hidden_layer = Dense(HIDDEN_LAYER_NEURONS,name='shared_hidden_layer')
    true_hidden = shared_hidden_layer(true_input)
    noisy_hidden = shared_hidden_layer(noisy_input)
    
    shared_output_layer = Dense(1, name='share_output_layer')
    true_context_score = shared_output_layer(true_hidden)
    noisy_context_score = shared_output_layer(noisy_hidden)
    loss_out = Lambda(customized_loss, output_shape=(1,), name='context_layer')([true_context_score, noisy_context_score])
    
#    score_hidden_layer = Dense(HIDDEN_LAYER_NEURONS,name='hidden_layer')
#    score_hidden = score_hidden_layer(true_input)
    
    score_output_layer = Dense(1, name='output_layer')
    score_output = score_output_layer(true_hidden)
#    print "printing output layer weights: " + str(score_output_layer.get_weights())
    model = Model(inputs=[true_sequence, noisy_sequence], outputs=[loss_out, score_output])
    
    return model

def load_word_embeddings():
    embeddings_index = dict()
    f = open('glove.6B.100d.txt')
    for line in f:
    	values = line.split()
    	word = values[0]
    	coefs = asarray(values[1:], dtype='float32')
    	embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))


train_corpus = []
test_corpus = []
valid_corpus = []
train_labels = []
test_labels = []
valid_labels = []

tokeniser = Tokenizer()
TRAIN_SEQ_TOTAL, VALID_SEQ_TOTAL, TEST_SEQ_TOTAL = get_true_sequences()
fp = open('/home/nam/Desktop/AspiringMinds/word_embeddings/train_true_sequences.txt', 'rb')
TRAIN_SEQ_TOTAL = NEGATIVE_SAMPLES*len(fp.readlines())
fp.close()
fp = open('/home/nam/Desktop/AspiringMinds/word_embeddings/valid_true_sequences.txt', 'rb')
VALID_SEQ_TOTAL = NEGATIVE_SAMPLES*len(fp.readlines())
fp.close()
embedding_matrix = load_word_embeddings()
model = CandW_model()

sgd = Adam(lr=3e-4)
model.compile(optimizer=sgd, loss={'context_layer': lambda y_true, y_pred:y_pred,'output_layer': 'mean_squared_error'}, loss_weights=[ALPHA, 1-ALPHA], metrics={'context_layer': 'accuracy', 'output_layer': 'accuracy'})
model.summary()
#print(model.metrics_names)
#weights = WeightPrinter()
train_checkpoint_file = '/home/nam/Desktop/AspiringMinds/dataset/train_weights.{epoch:02d}--{closs:.2f}.hdf5'
train_checkpoint = ModelCheckpoint(train_checkpoint_file, monitor='context_layer_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=5)
val_checkpoint_file = '/home/nam/Desktop/AspiringMinds/dataset/val_weights.{epoch:02d}-{val_closs:.2f}.hdf5'
val_checkpoint = ModelCheckpoint(val_checkpoint_file, monitor='val_context_layer_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=5)
history = model.fit_generator(train_generator(BATCH_SIZE), steps_per_epoch=TRAIN_SEQ_TOTAL//BATCH_SIZE, epochs=NUMBER_EPOCHS, validation_data=valid_generator(BATCH_SIZE), validation_steps=VALID_SEQ_TOTAL//BATCH_SIZE,verbose=1, callbacks=[val_checkpoint, train_checkpoint])
#model.fit({'true_sequence': X_true, 'noisy_sequence':X_noisy}, {'score_output':final_scores_d1})
print history.history['loss']
#scores = model.evaluate_generator(test_generator(BATCH_SIZE), steps=TEST_SEQ_TOTAL//BATCH_SIZE)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))

# serialize model to JSON
model_json = model.to_json()
with open("/home/nam/Desktop/AspiringMinds/word_embeddings/sswe_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/nam/Desktop/AspiringMinds/word_embeddings/sswe_model.h5")
print("Saved model to disk")