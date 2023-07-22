import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,  linear_model
from sklearn.metrics import mean_squared_error
#Linear regression using scikit-learn data
#Simple Linear regression - is the algo used when there are 1 dependent variable and 1 indepenedent variable.
# Multiple Linear Regression - Dependent variable is one and independent is more than one
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
# print(diabetes.DESCR)
print(diabetes_X)
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]
model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)
diabetes_Y_predicted = model.predict(diabetes_X_test)
print('Mean Squared Error:', mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))
print('Weight:', model.coef_)
print('Intercept', model.intercept_)
plt.scatter(diabetes_X_test, diabetes_Y_test)
plt.plot(diabetes_X_test, diabetes_Y_predicted)
plt.show()
