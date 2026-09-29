# Membaca nama (sebagai teks/string)
print('What is your name? ')
name = input()

# Membaca umur (dan mengubahnya menjadi integer)
print('What is your age? ')
age_str = input()
age = int(age_str)

# Membaca nilai/marks (dan mengubahnya menjadi float)
print('What are your marks? ')
marks_str = input()
marks = float(marks_str)

# Menampilkan hasilnya ke layar
print('Hello ' + name + "!")
print('Age: ' + str(age))
print('Marks: ' + str(marks))