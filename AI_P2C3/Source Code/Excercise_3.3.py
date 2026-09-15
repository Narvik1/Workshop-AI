# Penyelesaian Exercise 3.3
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')

# Memplot sepal length vs sepal width sebagai scatter plot
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.title('Scatter Plot: Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()