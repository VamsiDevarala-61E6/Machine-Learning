# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:34:52 2024

@author: vamsi
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data = pd.read_csv(r"C:\python_files\Online-Retail.csv",encoding='ISO-8859-1')

data['Description'] = le.fit_transform(data['Description'])
data['StockCode'] = le.fit_transform(data['StockCode'])
data['Country'] = le.fit_transform(data['Country'])
data['InvoiceNo'] = le.fit_transform(data['InvoiceNo'])
data.dropna(subset=['CustomerID'], inplace=True)
data.drop('InvoiceDate', axis=1, inplace=True)

x = data.values



from sklearn.cluster import KMeans
modelkm = KMeans(n_clusters=4) 
modelkm.fit(x)
centers = modelkm.cluster_centers_
labels = modelkm.predict(x)
print(set(labels))


