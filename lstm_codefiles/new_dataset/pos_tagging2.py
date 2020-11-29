#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:00:21 2018

@author: nam
"""
import os
import pandas as pd
import pickle
from os.path import expanduser 
from nltk.tag.stanford import StanfordPOSTagger
from nltk import word_tokenize, pos_tag
import enchant
import re

# The function stores the essays from all prompts and their respective grammar scores into two files
def get_essays_and_scores():
    root_dir = '/home/nam/Desktop/AspiringMinds/lstm_datafiles/dataset2/'
    store_file_corpus = '/home/nam/Desktop/AspiringMinds/lstm_datafiles/dataset2/essays.txt'
    store_file_scores = '/home/nam/Desktop/AspiringMinds/lstm_datafiles/dataset2/grammar_scores.txt'
    fp = open(store_file_corpus,'wb')
    wp = open(store_file_scores,'wb')
    engDic = enchant.Dict("en_GB")
    # Total 4399 essays and maximum score is 6.0
    file_name = 'data.csv'
    path = root_dir + str(file_name)
    print file_name
    pandas_data = pd.read_csv(path)
    essays = pandas_data['essay_text']
    grammar_scores1 = pandas_data['grammar_score']
    count = 0
    for essay, grammar_score1 in zip(essays, grammar_scores1):
        print count
        count = count + 1
        # Ignoring essays with negative scores. 
        if grammar_score1 < 0.0:
            continue
        #  Ignoring essays with all upper case letter. NLTK tags them wrongly.
        if str(essay).isupper() == True:
            continue
        essay = essay.split()
        # Checking for spelling errors and replacing them with the top word suggested by enchante
        for idx,word in enumerate(essay):
            if engDic.check(word) == True or len(engDic.suggest(word))==0:
                continue
            else:
                essay[idx] = engDic.suggest(word)[0]
        essay = ' '.join(essay)
        
        pickle.dump(essay,fp)
        pickle.dump(grammar_score1,wp)
    fp.close()
    wp.close()
    
def get_pos_tags():
#    # Setting up the Stanford POS tagger from NLTK
#    home = expanduser("~")
#    _path_to_model = home + '/stanford-postagger/models/english-bidirectional-distsim.tagger'
#    _path_to_jar = home + '/stanford-postagger/stanford-postagger.jar'
#    st = StanfordPOSTagger(model_filename=_path_to_model, path_to_jar=_path_to_jar, java_options='-mx10000m')
    
    # Loading the essays
    essays = []
    store_file_corpus = '/home/nam/Desktop/AspiringMinds/lstm_datafiles/dataset2/essays.txt'
    fp = open(store_file_corpus, 'rb')
    while True:
        try:
            essays.append(pickle.load(fp))
        except EOFError:
            break
    fp.close()
    fp = None
#    # To count the various length of essays.    
#    count_length_dict = dict()
#    for idx,essay in enumerate(essays):
#        print idx
#        essay = essay.split()
#        length = len(essay)
#        if length in count_length_dict:
#            count_length_dict[length] = count_length_dict[length] + 1
#        else:
#            count_length_dict[length] = 1
#
#    # Store all types of pos tags and their counts using stanford POS tagger
#    stanford_pos_dict = dict()
#    for idx,essay in enumerate(essays):
#        print idx
#        essay = essay.split()
#        tagged = st.tag(essay)
#        for pair_element in tagged:
#            word,pos = pair_element
#            if pos in stanford_pos_dict:
#                stanford_pos_dict[pos] = stanford_pos_dict[pos] + 1
#            else:
#                stanford_pos_dict[pos] = 1
#                
#    # Store all types of pos tags and their counts using TreeBank POS tagger
#    treebank_pos_dict = dict()
#    for idx,essay in enumerate(essays):
#        print idx
#        tagged = pos_tag(word_tokenize(essay))
#        for pair_element in tagged:
#            word,pos = pair_element
#            if pos in treebank_pos_dict:
#                treebank_pos_dict[pos] = treebank_pos_dict[pos] + 1
#            else:
#                treebank_pos_dict[pos] = 1
    
    # Preparing corpus. All essays are converted to their POS tags, and written to a file.
    new_corpus = list()
    fp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/dataset2/lstm_tagged_corpus.txt','w')
    for idx,essay in enumerate(essays):
        print idx
        tagged = pos_tag(word_tokenize(essay))
        new_essay = list()
        for pair_element in tagged:
            word, pos = pair_element
            new_essay.append(pos)
        new_essay = ' '.join(new_essay)
        new_corpus.append(new_essay)
        pickle.dump(new_essay,fp)
    fp.close()
    
#get_essays_and_scores()
get_pos_tags()