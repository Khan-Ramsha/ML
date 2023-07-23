import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("xyz.txt")
df.head(6)
x = df[['distance', 'years']]
y = df['price']
reg = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x.values, y, test_size=0.3, random_state=10)
reg.fit(x_train, y_train)
print("Predicted Price:", reg.predict([[350, 4]]))
print("Intercept: ", reg.intercept_)
print("Coefficient:", reg.coef_)
plt.scatter(df['distance'], df['years'], c=df['price'], cmap='viridis')
plt.xlabel(['distance', 'years'])
plt.ylabel(['price'])
plt.colorbar(label='Price')
plt.grid(True)
plt.show()
