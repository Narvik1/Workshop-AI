import numpy as np
import matplotlib.pyplot as plt

# 1. Membuat rentang nilai x, misalnya dari -10 hingga 10 sebanyak 100 titik
x = np.linspace(-10, 10, 100)

# 2. Mendefinisikan persamaan fungsi matematika
y1 = 3 * x + 4
y2 = 2 * x**2 + 1
y3 = x**3 + 9

# 3. Menggambar plot untuk setiap fungsi dengan warna dan label yang berbeda
plt.plot(x, y1, color='blue', label='y = 3x + 4')
plt.plot(x, y2, color='red', label='y = 2x^2 + 1')
plt.plot(x, y3, color='green', label='y = x^3 + 9')

# 4. Menambahkan judul dan label sumbu
plt.title('Plot Fungsi Matematika')
plt.xlabel('x')
plt.ylabel('y')

# 5. Menampilkan legenda grafik
plt.legend()

# 6. Menampilkan grafik ke layar
plt.show()