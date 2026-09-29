def sortarray(xs):
    n = len(xs)
    # Melakukan perulangan untuk membandingkan setiap elemen
    for i in range(n):
        for j in range(0, n - i - 1):
            # Jika elemen saat ini lebih besar dari elemen berikutnya, tukar posisinya
            if xs[j] > xs[j+1]:
                xs[j], xs[j+1] = xs[j+1], xs[j]
    return xs

# Mengacak urutan data untuk membuktikan fungsi berjalan dengan benar
data = [5, 2, 0, 4, 1, 3] 
t = sortarray(data)
print(t)