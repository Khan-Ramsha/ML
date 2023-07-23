import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_csv('def.txt')
df.head(10)
x = df[['age']]
y = df['results']
reg = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(x.values, y, test_size=0.3, random_state=10)
reg.fit(x_train, y_train)
print("predicted results:", reg.predict(x_test))
print("Score:", reg.score(x_test, y_test))
plt.scatter(df[['age']], df['results'], c=df['results'], cmap="viridis")
plt.xlabel('Age')
plt.ylabel('Results')
plt.colorbar(label='Results')
plt.grid(True)
plt.show()
