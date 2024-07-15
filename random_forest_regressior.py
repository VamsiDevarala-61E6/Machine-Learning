# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:51:42 2024

@author: vamsi
"""

import pandas as pd

data = pd.read_csv(r"C:\python_files\IMDB-Movie-Data.csv")

data['Revenue (Millions)'] = data['Revenue (Millions)'].fillna(data['Revenue (Millions)'].mean())
data['Metascore'] = data['Metascore'].fillna(data['Metascore'].mean())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

a = ['Title','Genre','Description','Director','Actors']

for i in a:
    data[i] = le.fit_transform(data[i])
    
    
x = data.iloc[ : , :-1].values
y = data.iloc[ : ,-1].values
    
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=(156))

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=10000)

model.fit(xtrain,ytrain)

from sklearn.metrics import r2_score

ypred = model.predict(xtest)

r2sc = r2_score(ytest, ypred)

print(r2sc*100)
print(f"{r2sc:.2}")




