#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:37:30 2020

@author: alexsantos
"""

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
from keras.utils import np_utils
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import time as time


# Step1: Create data set

base = pd.read_csv('challenge_raw_everyone.csv')

base_cleaned = base.dropna()
base['person'].value_counts()
base_cleaned['person'].value_counts()
# Step2: Getting Y and features


X = base_cleaned.iloc[:,0:15].values #ao colocar 1:2, o iloc retorna somente a coluna 1
y = base_cleaned.iloc[:,15:16].values #ao colocar 1:2, o iloc retorna somente a coluna 1
labelencoder = LabelEncoder()
classe_encoder = labelencoder.fit_transform(y)
base_cleaned['enconder']=classe_encoder
classe_dummy = np_utils.to_categorical(classe_encoder)


# Step2: Spliting in train and test files

X_train, X_test, y_train, y_test = train_test_split(X, classe_encoder, test_size=0.30)

scoring = ['accuracy', 'precision','f1','recall','roc_auc']

#Linear Regression
#lm = linear_model.LinearRegression()
#model = lm.fit(X_train, y_train)
#y_pred = lm.predict(X_test)
#accuracy_score(y_test, y_pred)


# Step 3: Fit a Decision Tree model as comparison
#Using cross-validation
clf = DecisionTreeClassifier()
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()
#getting time

start = time.time()
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
end = time.time()
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
tempo =[['Decision Tree',(end-start),media, desvio]]

#Usando Cross Validation








# Step 4: Fit a Random Forest model, " compared to "Decision Tree model, accuracy go up by 5%
clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()


start = time.time()
clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
end = time.time()
tempo.append(['Random Forest',(end-start)])






# Step 5: Fit a AdaBoost model, " compared to "Decision Tree model, accuracy go up by 10%
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)



clf = AdaBoostClassifier(n_estimators=100)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()





# Step 6: Fit a Gradient Boosting model, " compared to "Decision Tree model, accuracy go up by 10%
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)



clf = GradientBoostingClassifier(n_estimators=100)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()


#SVM

from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

start = time.time()
clf = LinearSVC(random_state=0, tol=1e-5, dual=False,)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
end = time.time()
print("Single SVC", end - start, clf.score(X,y))


start = time.time()
clf = LinearSVC(random_state=0, tol=1e-5, dual=False,)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()