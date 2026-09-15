# Example 3.22 - Q_test.py (Diperbarui)
import numpy as np
import pylab as plt
from Q_Utils import * # Import normal, pastikan file Q_Utils.py ada di folder yang sama

# Setting up Parameters =========================================
points_list = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (3, 6)]
goal = 6

# Show the routing graph
showgraph(points_list)

# The number of points of the R matrix
MATRIX_SIZE = 7

# create matrix R
R = createRmat(MATRIX_SIZE, points_list, goal)

# Create matrix Q
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# Learning parameter
gamma = 0.8

# Training ======================================================
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)

print("Score:", str(score), flush=True)
print("Trained Q matrix:", flush=True)
print(Q / np.max(Q) * 100, flush=True)

# Testing =======================================================
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

# Display Results ===============================================
print("Most efficient path:", flush=True)
print(steps, flush=True)

plt.figure()
plt.plot(scores)
plt.title("Convergence of Q-learning")
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.show()