import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("abc.txt")
df.head(10)
x = df[['distance']]
y = df['price']
plt.scatter(df['distance'], df['price'])
plt.xlabel('distance')
plt.ylabel('price')
plt.show()
reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
reg.fit(X_train, y_train)
print("predicted price", reg.predict([[70]]))
print("Intercept", reg.intercept_)
print("coef", reg.coef_)

