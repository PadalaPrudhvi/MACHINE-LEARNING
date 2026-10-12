import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:/Users/prudh/Downloads/emp_sal.csv")

x = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2].values


## linear regeression model

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# linear regression visualization

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# we will build polynomial build

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Level or Salary(Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

# svr model

from sklearn.svm import SVR
svr_reg = SVR()
svr_reg.fit(x, y)

svr_model_pred = svr_reg.predict([[6.5]])
svr_model_pred

print(svr_reg.get_params())

# KNN regressor

from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor()
knn_reg.fit(x,y)

#prediciton
knn_reg_pred = knn_reg.predict([[6.5]])
knn_reg_pred

#decision tree algorithm

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion = 'absolute_error',
                               splitter= 'random',
                               )
dt_reg.fit(x,y)

dt_reg_pred = dt_reg.predict([[6.5]])
dt_reg_pred

#random forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=45,
                               random_state=0,
                               max_depth=None,
                               max_samples=5,
                               min_samples_leaf=2
                               )
rf_reg.fit(x,y)
rf_reg_pred = rf_reg.predict([[6.5]])
rf_reg_pred

#xgboost regressior

import xgboost as xg 
xgb_r = xg.XGBRegressor(objective = 'reg:linear', n_estimators = 4)
xgb_r.fit(x,y)
xgb_reg_pred = xgb_r.predict([[6.5]])
xgb_reg_pred