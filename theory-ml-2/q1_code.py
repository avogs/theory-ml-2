import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient

n = 100
A = []
for i in range(-50, 0):
    A.append([i, -1])

for i in range(1, 51):
    A.append([i, 1])
    
A = np.array(A)

y = A[:, 1]
p = A[:, 0]

w = -1

# Part C:
ws = []
learning_rate = 0.001
for i in range(100):
    ws.append(w)
    gradient = np.sum((np.sign(w*p) - y)*p)/n
    w = w - learning_rate * gradient

ws.append(w)
    
plt.plot(ws)
plt.xlabel("iterations")
plt.ylabel("$\omega$")
plt.title("Gradient Descent Without Corrupted Labels")
plt.show()

# Part D:
for i in range(0,5):
    p[i] = p[i]*-1
    p[-(i + 1)] = p[-(i + 1)]*-1

ws = []
w = -1
learning_rate = 0.01
for i in range(100):
    ws.append(w)
    gradient = np.sum((np.sign(w*p) - y)*p)/n
    w = w - learning_rate * gradient

ws.append(w)

plt.plot(ws)
plt.xlabel("iterations")
plt.ylabel("$\omega$")
plt.title("Gradient Descent With Corrupted Labels")
plt.show()