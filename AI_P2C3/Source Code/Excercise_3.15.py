# Penyelesaian Exercise 3.15
import numpy as np
from sklearn.semi_supervised import LabelSpreading

# Menambahkan grup ketiga (Grup 3)
X = np.array([
    [0, 1], [1, 1], [2, 0], [3, 1],           # Grup 1 (4 titik)
    [10, 5], [11, 6], [12, 4], [13, 5],       # Grup 2 (4 titik)
    [20, 10], [21, 9], [22, 10], [23, 11]     # Grup 3 (4 titik tambahan)
])

# Menyiapkan array label sejumlah 12 titik dengan nilai awal -1
labels = np.full(12, -1.)

# Pastikan *hanya satu titik* di masing-masing grup yang memiliki label
labels[0] = 0  # Titik [0, 1] mewakili Grup 1
labels[4] = 1  # Titik [10, 5] mewakili Grup 2
labels[8] = 2  # Titik [20, 10] mewakili Grup 3

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)

output_labels = label_spread.transduction_
print("Label setelah penyebaran (Exercise 3.15):")
print(output_labels)