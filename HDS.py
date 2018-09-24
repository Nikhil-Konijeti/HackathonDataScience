#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import pandas as pd
import math as m
from __future__ import division

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, 20].values

dataset1=pd.read_csv('test.csv')
M = dataset1.iloc[:,1:21].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
M = sc.fit_transform(M)

# Applying LDa
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 5)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
M=lda.transform(M)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 44, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)
classifier.feature_importances_

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred1 = classifier.predict(M)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

s=int(m.sqrt(cm.size))
sum1=0
sum2=0  

for i in range(0,s):
    for j in range(0,s):
            if i==j:
                sum1 = sum1 + cm[i][j]
            else:
                sum2 = sum2 + cm[i][j]
                
total=sum1+sum2                
Accuracy=(sum1/total)*100            
print("The accuracy for the given test set is " + str(float(Accuracy)) + "%")

a = np.asarray(y_pred1)
np.savetxt("sub.csv", np.dstack((np.arange(1, a.size+1),a))[0],"%d,%d",header="id,price_range")