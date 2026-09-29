import math

# Meminta pengguna memasukkan daftar angka (dipisahkan oleh spasi)
print('Input a list of float numbers (separated by space): ')
user_input = input()

# Memecah teks input menjadi list string berdasarkan spasi
number_strings = user_input.split()

# Melakukan perulangan untuk setiap angka yang dimasukkan
for num_str in number_strings:
    # Mengubah teks menjadi angka desimal (float)
    x = float(num_str)
    
    # Menghitung nilai sinus menggunakan math.sin()
    y = math.sin(x)
    
    # Menampilkan hasil ke layar
    print("The sine of " + str(x) + " is " + str(y))