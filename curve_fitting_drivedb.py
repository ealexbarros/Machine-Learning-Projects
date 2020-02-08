#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 09:37:50 2020

@author: alexsantos
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


base_drive = pd.read_csv('drive1_1min.csv')
#base_drive = base_drive.iloc[1:19841,0:2].values
base_drive['Elapsed time']=np.linspace(1., 10, 119041)
#base['MLII'] = base['MLII'].astype(float)
x = base_drive.iloc[1:7201,2].values
y = base_drive.iloc[1:7201,1].values
y=y.astype(float)
plt.plot(x,y)
#plt.scatter(x,y)


z = np.polyfit(x, y, 5)
f = np.poly1d(z)
print(f)

# calculate new x's and y's
x_new = np.linspace(x[0], x[-1], 7200)
y_new = f(x_new)
plt.plot(x_new,y_new)
#plt.scatter(x,y)
y_fitted=y-y_new

plt.plot(x,y)
plt.plot(x_new,y_fitted)


