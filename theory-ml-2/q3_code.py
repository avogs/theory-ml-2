from os import close
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time
np.set_printoptions(suppress=True)

# Part C:
n = 500
m = 2*n

A = np.random.rand(m, n) * 2 - 1

eta = np.random.normal(loc = 0, scale = math.sqrt(0.5), size = m)
x_star = np.random.rand(n) * 2 - 1
b = np.matmul(A, x_star)
b = b + eta

distances = []

# Part D:
stepsize = 0.1
x = np.zeros(n)
for t in range(50):
    gradient = (np.dot(np.dot(2*x.T, A.T), A) - np.dot(2*b.T, A))/n
    x = x - stepsize * gradient
    distances.append(np.linalg.norm(x_star - x))

print("x_star:")
print(x_star)
print("x:")
print(x)
print("distance from x_star:")
print(np.linalg.norm(x_star - x))
plt.plot(distances)
plt.xlabel("iterations")
plt.ylabel("distance from x*")
plt.title("Gradient descent for linear regression")
#plt.show()

# Part E:
# pretty sure this isn't working based on how far off it is
closed_form_x_star = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))

print("x_star:")
print(x_star)
print("x:")
print(x_star - closed_form_x_star)
print("distance from x_star:")
print(np.linalg.norm(x_star - closed_form_x_star))

# Part E timing:

t0 = time.time()
for i in range(100):
    x = np.zeros(n)
    for t in range(50):
        gradient = (np.dot(np.dot(2*x.T, A.T), A) - np.dot(2*b.T, A))/n
        x = x - stepsize * gradient

t1 = time.time()

total = (t1 - t0) / 100
print("gradient descent time: ")
print(total)

t0 = time.time()
for i in range(100):
    closed_form_x_star = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))

t1 = time.time()

total = (t1 - t0) / 100
print("closed form time: ")
print(total)