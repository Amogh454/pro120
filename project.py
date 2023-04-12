import pandas as pd
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
from six import StringIO
from IPython.display import Image

import pydotplus
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

column_names = ["glucose","bloodpressure"]
income = ['diabetes']
data = pd.read_csv('data2.csv')

X = data[column_names]
y = data[income]

Xtrain1, Xtest1, Ytrain1, Ytest1 = train_test_split(X, y, test_size=0.5, random_state=42)
Xtrain2, Xtest2, Ytrain2, Ytest2 = train_test_split(X, y, test_size=0.5, random_state=42)

sc = StandardScaler()


Xtrain1 = sc.fit_transform(Xtrain1)
Xtest1 = sc.fit_transform(Xtest1)

model1 = GaussianNB()
model1.fit(Xtrain1, Ytrain1)
yprediction1 = model1.predict(Xtest1)
accuracy1 = accuracy_score(Ytest1, yprediction1)
print('Accuracy of model 1:', accuracy1)

Xtrain2 = sc.fit_transform(Xtrain2)
Xtest2 = sc.fit_transform(Xtest2)

model2 = LogisticRegression()
model2.fit(Xtrain2, Ytrain2)
yprediction2 = model2.predict(Xtest2)
accuracy2 = accuracy_score(Ytest2, yprediction2 )
print('Accuracy of model 2:', accuracy2 )
