import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv(r"C:\Users\prudh\Downloads\Churn_Modelling.csv")
# split with IV&DV
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

#Label encoding the "Gender column"
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)


#One hot encoding the "Geography" column

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
                        remainder='passthrough')

X = np.array(ct.fit_transform(X))


#splitting the dataset into the Trainig set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)


#Training XGBOOST on the Trainig set
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators=25)
classifier.fit(X_train, y_train)

print(classifier.get_params())

#predicitng the test set results
y_pred = classifier.predict(X_test)

#making the confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


bias = classifier.score(X_train,y_train)
bias
