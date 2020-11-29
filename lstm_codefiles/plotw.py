#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:56:52 2018

@author: nam
"""

import matplotlib.pyplot as plt
import pickle, math
import numpy as np
fp = open('/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_train_scores.txt','rb')

X = []
while True:
        try:
            X.append(pickle.load(fp))
        except EOFError:
            break
X =  [x for x in X]
line = np.linspace(1,len(X), len(X))
plt.hist(X)
plt.show()
 