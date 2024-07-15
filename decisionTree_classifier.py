# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:29:57 2024

@author: vamsi
"""

import pandas as pd

data = pd.read_csv(r"C:\python_files\iris.csv")

data = data.drop(["color"],axis=1)

data["pl"] = data['pl'].fillna(data['pl'].mean())

x = data.iloc[ : , :4].values
y = data.iloc[ : , -1].values

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.25,random_state=(10))

from sklearn.tree import DecisionTreeClassifier

modeldtc = DecisionTreeClassifier()

modeldtc = modeldtc.fit(xtrain,ytrain)

from sklearn.metrics import accuracy_score

ypred = modeldtc.predict(xtest)

ac = accuracy_score(ytest, ypred)

print(ac*100)