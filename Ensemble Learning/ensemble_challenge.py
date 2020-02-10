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

base = pd.read_csv('challenge_raw_everyone.csv')
original_columns = ['mean_q', 'mean_r', 'mean_s', 'mean_p' , 'mean_t','stdev_q', 'stdev_r','stdev_s',
                   'mean_rr_interval', 'mean_rq_amplitude', 'mean_qrs_interval', 'mean_qs_distance',
                   'mean_qt_distance', 	'mean_qrs_offset',	'mean_qrs_onset']
		
X = base[original_columns]
X = X.apply(lambda x: x.fillna(x.mean()))

#X = base.iloc[:,0:15].values #ao colocar 1:2, o iloc retorna somente a coluna 1
#base_cleaned['person'].value_counts()
#base_cleaned['person'].value_counts()
#base_cleaned_n['person'].value_counts()
# Step2: Getting Y and features
#base.loc[pd.isna(base['mean_rr_interval'])]
#base['mean_rr_interval'].value_counts()
#base.loc[pd.isna(base['mean_rr_interval'])]
#base_n['person'].value_counts()
y = base.iloc[:,15:16].values #ao colocar 1:2, o iloc retorna somente a coluna 1


import pandas as pd 
  
# initialise data of lists. 
data = {'Time':[], 'Mean':[],'Std':[],'Acc':[],'F1':[],'Recall':[],'Precision':[]} 
  
# Creates pandas DataFrame. 
df = pd.DataFrame(data, index =['DT', 'RF', 'ExtraT', 'AdaB','GradientB','Voting','NeuralN']) 

# Step2: Spliting in train and test files
base_nova=base.head(11600)
base_nova['person'].value_counts()
base_nova = base_nova.dropna()
X = base_nova.iloc[:,0:15].values #ao colocar 1:2, o iloc retorna somente a coluna 1
y = base_nova.iloc[:,15:16].values #ao colocar 1:2, o iloc retorna somente a coluna 1
#labelencoder = LabelEncoder()
#classe_encoder = labelencoder.fit_transform(y)
#base_nova['enconder']=classe_encoder
#classe_dummy = np_utils.to_categorical(classe_encoder)
#base_nova['enconder'].value_counts




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

plt.figure(figsize=(20,10))
plt.title("Cumulative distribution of number of examples")
base['person'].value_counts(ascending=True).hist(cumulative=True, density=1)


list_of_training_x = np.array_split(X_train, 7)
list_of_training_y = np.array_split(y_train, 7)
n_estimators=60

rf_step_2 = RandomForestClassifier(warm_start=True, n_estimators=n_estimators, max_depth=100, min_samples_leaf=3, min_samples_split=10, verbose=3)
for i in range(7):
    rf_step_2.fit(list_of_training_x[i], list_of_training_y[i])
    rf_step_2.set_params(n_estimators=n_estimators)
    n_estimators+=20

predictions_step_2 = rf_step_2.predict(X_test)

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
metrics_DT =[['Decision Tree',(end-start),media, desvio,acc,f1,recall,precision]]
print(metrics)


# Step 3: Fit a Random Forest Model model as comparison
#Using cross-validation
clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
resultados = cross_val_score(estimator = clf,
                             X = X, y = classe_encoder,
                             cv = cv, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()
print(resultados)
print(media,desvio)


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
metrics_RF=['Random Forest',(end-start),media, desvio,acc,f1,recall,precision]
print(metrics_RF)

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
metrics_Ext=['Extra Trees Classifier',(end-start),media, desvio,acc,f1,recall,precision]
print(metrics_Ext)

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
metrics_ada=['AdaBoost',(end-start),media, desvio,acc,f1,recall,precision]
print(metrics_ada)

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
metrics_Grad=['GradientBoosting',(end-start),media, desvio,acc,f1,recall,precision]
print(metrics_Grad)


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
metrics_SVM=['SVM',(end-start),media, desvio,acc,f1,recall,precision]
print(metrics_SVM)


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
classificador.add(Dense(units = 200, activation = 'relu', input_dim = 15))
classificador.add(Dense(units = 200, activation = 'relu'))
classificador.add(Dense(units = 200, activation = 'relu'))
classificador.add(Dense(units = 200, activation = 'relu'))
classificador.add(Dense(units = 200, activation = 'relu'))
classificador.add(Dense(units = 401, activation = 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 1000,
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



