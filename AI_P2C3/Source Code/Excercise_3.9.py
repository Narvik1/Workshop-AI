# Penyelesaian Exercise 3.9
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

reg = RandomForestRegressor(random_state=0)
reg.fit(X_train, y_train)
score = reg.score(X_test, y_test) # R-squared untuk regresi
print("Skor R-squared (Random Forest Regressor) pada data diabetes: %.2f" % score)