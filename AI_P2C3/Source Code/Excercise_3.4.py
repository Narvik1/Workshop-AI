# Penyelesaian Exercise 3.4
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
# Mengonversi data ke DataFrame agar lebih mudah diplot
df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Memilih fitur spesifik sesuai instruksi: radius, area (size), texture, smoothness
features_to_plot = ['mean radius', 'mean area', 'mean texture', 'mean smoothness']

# Membuat histogram
df_cancer[features_to_plot].hist(figsize=(10, 8), bins=20)
plt.suptitle('Histogram Fitur Breast Cancer')
plt.show()