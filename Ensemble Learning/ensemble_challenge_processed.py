#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:10:47 2020

@author: alexsantos
"""

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
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn import metrics
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score,f1_score, recall_score,precision_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import time as time
from sklearn.metrics import make_scorer
from sklearn.svm import LinearSVC
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,ShuffleSplit
from keras.utils import np_utils
from sklearn.ensemble import VotingClassifier

# Step1: Create data set

base = pd.read_csv('challenge_processed_everyone.csv')
base_cleaned = base.dropna()
#base_cleaned['person'].value_counts()
#base_cleaned['person'].value_counts()
#base_cleaned_n['person'].value_counts()
# Step2: Getting Y and features
#base.loc[pd.isna(base['mean_rr_interval'])]
#base['mean_rr_interval'].value_counts()
valores={'mean_rr_interval':930}
base=base.fillna(value=valores)
#base.loc[pd.isna(base['mean_rr_interval'])]
#base_n['person'].value_counts()
base_cleaned['person'].value_counts

i=base.loc[base.person=='tr14-0291']
base=base_cleaned
X = base.iloc[:,0:15].values #ao colocar 1:2, o iloc retorna somente a coluna 1
y = base.iloc[:,15:16].values #ao colocar 1:2, o iloc retorna somente a coluna 1
labelencoder = LabelEncoder()
classe_encoder = labelencoder.fit_transform(y)
#base['enconder']=classe_encoder
classe_dummy = np_utils.to_categorical(classe_encoder)

# Step2: Spliting in train and test files

X_train, X_test, y_train, y_test = train_test_split(X, classe_encoder, test_size=0.30)


#Linear Regression
#lm = linear_model.LinearRegression()
#model = lm.fit(X_train, y_train)
#y_pred = lm.predict(X_test)
#accuracy_score(y_test, y_pred)


# Step 3: Fit a Decision Tree model as comparison
#Using cross-validation
clf = DecisionTreeClassifier()
cv = ShuffleSplit(n_splits=10, random_state=0)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = cv, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()
#getting metrics

start = time.time()
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
end = time.time()
y_pred = clf.predict(X_test)
acc=accuracy_score(y_test, y_pred)
f1=f1_score(y_test, y_pred, average='macro')
recall=recall_score(y_test, y_pred, average='macro')
precision=precision_score(y_test, y_pred, average='macro')
#roc=roc_auc_score(y_test, y_pred)
metrics =[['Decision Tree',(end-start),media, desvio,acc,f1,recall,precision]]

# Step 3: Fit a Random Forest Model model as comparison
#Using cross-validation
clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = cv, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()

#getting metrics
start = time.time()
clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
clf.fit(X_train, y_train)
end = time.time()
y_pred = clf.predict(X_test)
acc=accuracy_score(y_test, y_pred)
f1=f1_score(y_test, y_pred, average='macro')
recall=recall_score(y_test, y_pred, average='macro')
precision=precision_score(y_test, y_pred, average='macro')
#roc=roc_auc_score(y_test, y_pred,multi_class='ovr')
metrics.append(['Random Forest',(end-start),media, desvio,acc,f1,recall,precision])

# Step 3: Fit a ExtraTreesClassifier Model model as comparison
#Using cross-validation
clf = ExtraTreesClassifier(n_estimators=100, max_features="auto",random_state=0)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = cv, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()

#getting metrics
start = time.time()
clf = ExtraTreesClassifier(n_estimators=100, max_features="auto",random_state=0)
clf.fit(X_train, y_train)
end = time.time()
y_pred = clf.predict(X_test)
acc=accuracy_score(y_test, y_pred)
f1=f1_score(y_test, y_pred, average='macro')
recall=recall_score(y_test, y_pred, average='macro')
precision=precision_score(y_test, y_pred, average='macro')
#roc=roc_auc_score(y_test, y_pred,multi_class='ovr')
metrics.append(['Extra Trees Classifier',(end-start),media, desvio,acc,f1,recall,precision])


# Step 4: Fit a AdaBoost model,
clf = AdaBoostClassifier(n_estimators=100)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = cv, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()

#getting metrics
start = time.time()
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
end = time.time()
y_pred = clf.predict(X_test)
acc=accuracy_score(y_test, y_pred)
f1=f1_score(y_test, y_pred, average='macro')
recall=recall_score(y_test, y_pred, average='macro')
precision=precision_score(y_test, y_pred, average='macro')
#roc=roc_auc_score(y_test, y_pred,multi_class='ovr')
metrics.append(['AdaBoost',(end-start),media, desvio,acc,f1,recall,precision])


# Step 5: Fit a Gradient Boosting model, " compared to "Decision Tree model, accuracy go up by 10%
clf = GradientBoostingClassifier(n_estimators=100)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = cv, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()

#getting metrics
start = time.time()
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
end = time.time()
y_pred = clf.predict(X_test)
acc=accuracy_score(y_test, y_pred)
f1=f1_score(y_test, y_pred, average='macro')
recall=recall_score(y_test, y_pred, average='macro')
precision=precision_score(y_test, y_pred, average='macro')
#roc=roc_auc_score(y_test, y_pred,multi_class='ovr')
metrics.append(['GradientBoosting',(end-start),media, desvio,acc,f1,recall,precision])



#SVM
# Step 6: Fit a Linear SVM
clf = LinearSVC(random_state=0, tol=1e-5, dual=False,)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = cv, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()

#getting metrics
start = time.time()
clf = LinearSVC(random_state=0, tol=1e-5, dual=False,)
clf.fit(X_train, y_train)
end = time.time()
y_pred = clf.predict(X_test)
acc=accuracy_score(y_test, y_pred)
f1=f1_score(y_test, y_pred, average='macro')
recall=recall_score(y_test, y_pred, average='macro')
precision=precision_score(y_test, y_pred, average='macro')
#roc=roc_auc_score(y_test, y_pred,multi_class='ovr')
metrics.append(['SVM',(end-start),media, desvio,acc,f1,recall,precision])

#step 7: votingrnd_clf = RandomForestClassifier()
metrics2=[]
dt_clf = DecisionTreeClassifier()
rnd_clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
ext_clf = ExtraTreesClassifier(n_estimators=100, max_features="auto",random_state=0)
voting_clf = VotingClassifier(
        estimators=[('lr', dt_clf), ('rf', rnd_clf), ('ext', ext_clf)],
        voting='hard')
voting_clf.fit(X_train, y_train)
for clf in (dt_clf, rnd_clf, ext_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics2.append(accuracy_score(y_test, y_pred))
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))



# Step 7: Fit a Neural Network

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe_encoder = labelencoder.fit_transform(y)
classe_dummy = np_utils.to_categorical(classe_encoder)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(X, classe_dummy, test_size=0.3)
#input_dim=atributos previsores
start = time.time()
classificador = Sequential()
classificador.add(Dense(units = 1000, activation = 'relu', input_dim = 15))
classificador.add(Dense(units = 1000, activation = 'relu'))
classificador.add(Dense(units = 1000, activation = 'relu'))
classificador.add(Dense(units = 1000, activation = 'relu'))
classificador.add(Dense(units = 1000, activation = 'relu'))
classificador.add(Dense(units = 1983, activation = 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 2000,
                  epochs = 1000)
end = time.time()
resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes_original=previsoes
previsoes = (previsoes > 0.5)
print(end-start)
import numpy as np
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes2, classe_teste2)



