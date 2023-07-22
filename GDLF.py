#Gradient Descent - Full dataSet(Takes all data at once)
#MiniBatch - Where u can choose ur choice of number o dataset either 100 of 1 million.Choosen by data scientist
#Stochastic Gradient Descent - Only one data at once.

# Gradient Descent and Loss Function
# Formula :
# MSE(Mean Squared Error) = 1/n sum of (Y1-YP) ^2 Y1 is actual value and YP is predicted one.
# Y = mx+c.
# m = m - LR * PD (m)
# c = c - LR * PD (c)
# MSE = 1/n sumation of (Y1-(mx1+c))^2
# PD (m) = -2/n sumation of x1 (y1-(mx1+b))
# PD (c) = -2/n sumation of (y1-(mx1+b))
# Now Let's Start Coding of Gradient Descent
import numpy as np


def gradient_descent(x, y):
    m = c = 0
    rate = 0.001
    iteration = 6500
    n = len(x)
    for i in range(iteration):
        y_predicted = m * x + c
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        mp = -(2 / n) * sum(x * (y - y_predicted))
        cp = -(2 / n) * sum(y - y_predicted)
        m = m - rate * mp
        c = c - rate * cp
        print("m:", m)
        print("n:", n)
        print("cost:", cost)
        print("\n")


x = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
gradient_descent(x, y)
