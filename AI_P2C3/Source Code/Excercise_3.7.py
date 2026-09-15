# Penyelesaian Exercise 3.7
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import load_breast_cancer

# Muat data breast cancer
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Lakukan PCA untuk mereduksi menjadi 2 komponen utama (agar mudah diplot 2D)
pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot Data PCA
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA pada Dataset Breast Cancer')
plt.show()