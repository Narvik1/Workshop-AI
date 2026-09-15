# Penyelesaian Exercise 3.10
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

names = ["SVM Regressor", "Decision Tree", "Random Forest", "Nearest Neighbors"]
regressors = [SVR(), DecisionTreeRegressor(), RandomForestRegressor(), KNeighborsRegressor()]

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

for name, reg in zip(names, regressors):
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    print(name + ": " + str(score))