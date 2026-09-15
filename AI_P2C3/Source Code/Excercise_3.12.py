# Penyelesaian Exercise 3.12
from sklearn import linear_model
from sklearn.datasets import load_linnerud

# Muat data Linnerrud (fisiologis memprediksi atribut olahraga)
linnerrud = load_linnerud()
X = linnerrud.data
y = linnerrud.target # Multiple output

reg = linear_model.LinearRegression()
reg.fit(X, y)

print('Coefficients:\n', reg.coef_)
print('Intercept:\n', reg.intercept_)