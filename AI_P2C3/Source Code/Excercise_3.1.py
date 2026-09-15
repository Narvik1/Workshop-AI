# Penyelesaian Exercise 3.1
from sklearn import svm

# Menambahkan 2 sampel baru sehingga total menjadi 6 sampel
# Format: Height [cm], Weight [kg], Shoesize [UK]
X = [
    [170, 70, 10], 
    [180, 80, 12], 
    [170, 65, 8],  
    [160, 55, 7],
    [175, 73, 10], # Tambahan 1 (Misal: Laki-laki)
    [155, 48, 5]   # Tambahan 2 (Misal: Perempuan)
]

# Gender, 0: Male, 1: Female
y = [0, 0, 1, 1, 0, 1] # Menambahkan label 0 dan 1 untuk sampel baru

clf = svm.SVC()
clf.fit(X, y)

#Predict
p = clf.predict([[160, 60, 7]])
print ("Hasil prediksi:", p)