import random
import math
import matplotlib.pyplot as plt

# initialize all points
n = 100
points = []
for i in range(n):
    points.append((i / n, -1))
for i in range(n):
    points.append((i / n, 1))

# set initial values for x and y
xt = 1
yt = 1
learning_rate = 0.1
yt_s = []

for t in range(200):
    rand = random.randint(0, 199)
    xt = xt - learning_rate*(2*xt - 2*points[rand][0])
    yt = yt - learning_rate*(2*yt - 2*points[rand][1])
    yt_s.append(yt)

print("y_t vals for constant learning rate: ")
for y_t in yt_s:
    print(y_t)

plt.plot(yt_s)
plt.xlabel("iteration number (t)")
plt.ylabel("$y_t$ value")
plt.title("Learning Rate = 0.1")
plt.show()

    
# reset initial values for x and y
xt = 1
yt = 1
yt_s2 = []

for t in range(200):
    rand = random.randint(0, 199)
    xt = xt - (learning_rate/(t + 1))*(2*xt - 2*points[rand][0])
    yt = yt - (learning_rate/(t + 1))*(2*yt - 2*points[rand][1])
    yt_s2.append(yt)
    
print("y_t vals for learning rate divided by t: ")
for y_t in yt_s2:
    print(y_t)

plt.plot(yt_s2)
plt.xlabel("iteration number (t)")
plt.ylabel("$y_t$ value")
plt.title("Learning Rate = 0.1/t + 1")
plt.show()
    
# reset initial values for x and y
xt = 1
yt = 1
yt_s3 = []

for t in range(200):
    rand = random.randint(0, 199)
    xt = xt - (learning_rate/math.sqrt(t + 1))*(2*xt - 2*points[rand][0])
    yt = yt - (learning_rate/math.sqrt(t + 1))*(2*yt - 2*points[rand][1])
    yt_s3.append(yt)
    
print("y_t vals for learning rate divided by sqrt(t): ")
for y_t in yt_s3:
    print(y_t)
    
plt.plot(yt_s3)
plt.xlabel("iteration number (t)")
plt.ylabel("$y_t$ value")
plt.title("Learning Rate = 0.1/$\sqrt{t + 1}$")
plt.show()