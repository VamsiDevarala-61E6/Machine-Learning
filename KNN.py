# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:23:57 2024

@author: vamsi
"""
import pandas as pd

data = pd.read_csv(r"C:\python_files\iris.csv")

x = data.iloc[ : , :4].values
y = data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split
xtest,xtrain,ytest,ytrain = train_test_split(x,y,test_size=0.3,random_state=(77))

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=6)

model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score

ac = accuracy_score(ytest, ypred)

print(ac*100)

