# Exercise 3.16 - Q-Learning dengan 8 State
import numpy as np
import pylab as plt
from Q_Utils import * # Pastikan file Q_Utils.py ada di folder yang sama

# ===============================================================
# 1. SETUP PARAMETER & ROUTING DIAGRAM (8 STATE)
# ===============================================================
# Desain rute (Edge) antar titik (Node)
# 0: Drop Zone        4: Gudang Amunisi
# 1: Perimeter Luar   5: Pos Komando
# 2: Barak Pasukan    6: Pintu Bunker
# 3: Stasiun Radar    7: Target Utama (Goal)

points_list = [
    (0, 1), 
    (1, 2), (1, 3), 
    (2, 4), (2, 5), 
    (3, 5), (3, 6), 
    (4, 7), (5, 7), (6, 7)
]

goal = 7              # Target state adalah 7
MATRIX_SIZE = 8       # Total ada 8 state (0 sampai 7)

# Tampilkan graf rute
showgraph(points_list)

# Buat matriks Reward (R)
R = createRmat(MATRIX_SIZE, points_list, goal)

# Buat matriks Q (inisialisasi dengan 0)
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# Parameter Pembelajaran
gamma = 0.8

# ===============================================================
# 2. TRAINING
# ===============================================================
print("Memulai proses training...")
scores = []
for i in range(1000): # Iterasi dinaikkan sedikit agar lebih stabil untuk 8 state
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)

print("\nTraining selesai!")
print("Trained Q matrix:")
print(np.round(Q / np.max(Q) * 100, 1)) # Dibulatkan agar lebih rapi saat dicetak

# ===============================================================
# 3. TESTING (Mencari Rute Tercepat)
# ===============================================================
current_state = 0
steps = [current_state]

while current_state != goal:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index))
    else:
        next_step_index = int(next_step_index[0])
    
    steps.append(next_step_index)
    current_state = next_step_index

# ===============================================================
# 4. DISPLAY RESULTS
# ===============================================================
print("\n===============================")
print("Rute Paling Efisien (0 -> 7):")
print(steps)
print("===============================\n")

# Plot konvergensi Q-Learning
plt.figure(figsize=(10, 5))
plt.plot(scores, color='green')
plt.title("Konvergensi Q-Learning (8-State Routing)")
plt.xlabel("Iterasi")
plt.ylabel("Skor")
plt.grid(True)
plt.show()