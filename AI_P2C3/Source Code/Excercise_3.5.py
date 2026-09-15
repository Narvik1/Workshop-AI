# Penyelesaian Exercise 3.5
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import joblib # Menggunakan joblib untuk menyimpan dan memuat model

X, y = load_iris(return_X_y=True)

# 1. Melatih Model
clf = GaussianNB()
clf.fit(X, y)

# 2. Menyimpan model ke sebuah file (Serialization)
joblib.dump(clf, 'naive_bayes_model.pkl')
print("Model berhasil disimpan ke 'naive_bayes_model.pkl'")

# 3. Memuat model dari file (Deserialization)
clf_loaded = joblib.load('naive_bayes_model.pkl')

# 4. Membuat prediksi dengan model yang telah dimuat
p = clf_loaded.predict([[5.0, 3.4, 1.5, 0.4]])
print("Hasil prediksi dari model yang dimuat:", p)