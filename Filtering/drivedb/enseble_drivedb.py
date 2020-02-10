#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 11:54:29 2020

@author: alexsantos
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


base_drive = pd.read_csv('drivedb_final1.csv')

base_cleaned = base_drive.dropna()
base_drive['person'].value_counts()
base_cleaned['person'].value_counts()


from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
# Step1: Create data set
from keras.utils import np_utils
X, y = make_moons(n_samples=10000, noise=.5, random_state=0)
# Step2: Split the training test set


X = base_cleaned.iloc[:,0:10].values #ao colocar 1:2, o iloc retorna somente a coluna 1
y = base_cleaned.iloc[:,10:11].values #ao colocar 1:2, o iloc retorna somente a coluna 1


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe_encoder = labelencoder.fit_transform(y)


classe_dummy = np_utils.to_categorical(classe_encoder)


np.where(array1==0, 1, array1) 


X, y = make_moons(n_samples=10000, noise=.5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, classe_encoder, test_size=0.30)
# Step 3: Fit a Decision Tree model as comparison
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Step 4: Fit a Random Forest model, " compared to "Decision Tree model, accuracy go up by 5%
clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Step 5: Fit a AdaBoost model, " compared to "Decision Tree model, accuracy go up by 10%
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Step 6: Fit a Gradient Boosting model, " compared to "Decision Tree model, accuracy go up by 10%
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)