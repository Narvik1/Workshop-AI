# Penyelesaian Exercise 3.11
import matplotlib.pyplot as plt
from scipy import stats

# Menambah titik data
x = [0, 1, 2, 3, 4, 5, 6, 7]
y = [3, 5, 5, 6, 7, 8, 9, 10]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x_val):
    return slope * x_val + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y, label='Data Aktual', color='blue')
plt.plot(x, mymodel, label='Regresi Linear', color='red')
plt.xlabel('Nilai X')
plt.ylabel('Nilai Y')
plt.title('Regresi Linear dengan Data Tambahan')
plt.legend()
plt.grid(True)
plt.show()