import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\prudh\Downloads\logit classification.csv")

# split with IV&DV
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# 80-20 TRAIN THE SPLIT

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.20,
                                                    random_state=0)


# SCALE THE DATA
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''


# TRAIN THE MODEL X_TRAIN AND Y_TRAIN
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# create predicition for x_test
y_pred = classifier.predict(X_test)

# confusion matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#This is to get the model accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

# This is get the classification Report
from sklearn.metrics import classification_report 
cr = classification_report(y_test, y_pred)
cr


bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance



#future prediciton on validation data
dataset1 = pd.read_csv(r"C:\Users\prudh\Downloads\final1.csv")
d2 = dataset1.copy()
dataset1 = dataset1.iloc[:, [3, 4]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)


y_pred1 = pd.DataFrame()

d2 ['y_pred1'] = classifier.predict(M)

d2.to_csv('final.csv')







