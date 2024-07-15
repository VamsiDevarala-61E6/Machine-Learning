# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:34:52 2024

@author: vamsi
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv(r"C:\Users\vamsi\Downloads\CarPrice_Assignment.csv")

le = LabelEncoder()

a = ['CarName','carbody','drivewheel','enginetype','fuelsystem']

for i in data:
    if i in a:
        data[i] = le.fit_transform(data[i])
        



x = data.iloc[ : , :-1].values
y = data.iloc[ : , -1].values



from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.25,random_state=(5))

from sklearn.tree import DecisionTreeRegressor

modeldr = DecisionTreeRegressor()





modeldr = modeldr.fit(xtrain,ytrain)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ypred = modeldr.predict(xtest)

mse = mean_squared_error(ytest, ypred)
mae = mean_absolute_error(ytest, ypred)
r2 = r2_score(ytest, ypred)

print(mse)
print(mae)
print(r2*100)