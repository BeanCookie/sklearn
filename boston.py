from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_Y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_Y)

# print(model.coef_)
# print(model.intercept_)

print(model.score(data_X, data_Y))

print(model.predict(data_X[:4, :]))
print(data_Y[:4])


X, Y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)

plt.scatter(X, Y)
plt.show()