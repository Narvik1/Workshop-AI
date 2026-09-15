# Penyelesaian Exercise 3.14
import numpy as np
from sklearn.semi_supervised import LabelSpreading

# Menambahkan 2 titik baru ke masing-masing grup
# Grup 1 ditambah: [-1, 0] dan [4, 1]
# Grup 2 ditambah: [9, 5] dan [14, 4]
X = np.array([
    [-1, 0], [0, 1], [1, 1], [2, 0], [3, 1], [4, 1],       # Grup 1 (Total 6 titik)
    [9, 5], [10, 5], [11, 6], [12, 4], [13, 5], [14, 4]    # Grup 2 (Total 6 titik)
])

# Menyiapkan array label sejumlah 12 titik dengan nilai awal -1 (tidak berlabel)
labels = np.full(12, -1.)

# Kita beri label pada satu titik di Grup 1 dan satu titik di Grup 2
labels[1] = 0   # Titik [0, 1] dari Grup 1 diberi label 0
labels[-2] = 1  # Titik [13, 5] dari Grup 2 diberi label 1

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)

output_labels = label_spread.transduction_
print("Label setelah penyebaran (Exercise 3.14):")
print(output_labels)