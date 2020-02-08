#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 10:03:53 2020

@author: alexsantos
"""


import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils
import matplotlib.pyplot as plt
import scipy.signal.medfilt as medfilt

base = pd.read_csv('mitdb_100_.csv')
#x = base.iloc[0:720,0].values
base['Elapsed time']=np.linspace(1., 2, 3601)
#base['MLII'] = base['MLII'].astype(float)
x = base.iloc[1:120,2].values
y = base.iloc[1:120,1].values
y=y.astype(float)
plt.plot(x,y)
baseline1 = medfilt(y,k=3) 
plt.plot(x,baseline1)
