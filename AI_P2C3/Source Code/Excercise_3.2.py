# Penyelesaian Exercise 3.2
from sklearn import svm, datasets

iris = datasets.load_iris()
# Mengubah agar mengambil fitur ketiga dan keempat (Petal length dan Petal width)
X = iris.data[:, 2:4] 
y = iris.target
#0: Setosa, 1: Versicolour, 2: Virginica

clf = svm.SVC()
clf.fit(X, y)

# Memprediksi bunga menggunakan data petal length dan petal width (contoh: 1.5, 0.2)
p = clf.predict([[1.5, 0.2]]) 
print ("Hasil prediksi berdasarkan petal:", p)