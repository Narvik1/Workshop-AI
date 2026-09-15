# Penyelesaian Exercise 3.6
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Mengubah n_samples menjadi 2000 dan n_features menjadi 6
X, y = make_classification(n_samples=2000, n_features=6, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

# Karena ada 6 fitur, input prediksi juga harus memiliki 6 nilai
print("Hasil prediksi LDA (2000 sampel, 6 fitur):", clf.predict([[0, 0, 0, 0, 0, 0]]))