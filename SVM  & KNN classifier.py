import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\prudh\Downloads\logit classification.csv")


X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.20,
                                                    random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Trainig the SVM model on the Trainig set

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)


# predicting the Test set results
y_pred = classifier.predict(X_test)

#making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#This is to get models accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


bias = classifier.score(X_train,y_train)
print(bias)

variance = classifier.score(X_test,y_test)
print(variance)

# This is get the classification Report
from sklearn.metrics import classification_report 
cr = classification_report(y_test, y_pred)
cr

#knn classifier algorthim
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier 
classifier_knn.fit(X_train, y_train)

#predicting the Test set results
y_pred = classifier_knn.predict(X_test)






