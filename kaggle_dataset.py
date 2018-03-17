# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 11:34:44 2018

@author: Rinky
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('train.csv')
dataset2= pd.read_csv('test.csv')

dataset = dataset.dropna()
dataset2 = dataset2.dropna()


X_train = dataset.iloc[:,[0]]
y_train= dataset.iloc[:,[1]]
X_test= dataset2.iloc[:,[0]]
y_test = dataset2.iloc[:,[1]]

#fitting simple linear model onto training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set salaries
y_pred = regressor.predict(X_test)


#graphs for training data
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#graphs for test data
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_test,regressor.predict(X_test),color = 'green')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()