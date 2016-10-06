import numpy as np
import matplotlib.pyplot as plt
import math


def linreg(x, y):
    w1 = (len(x) * (np.sum(x*y)) - (np.sum(x)) * (np.sum(y))) / (len(x) * np.sum(x**2) - (np.sum(x))**2)
    w0 = (np.sum(y) - w1 * np.sum(x)) / len(x)
    return [w0, w1]


def sigm(x):
    return 1 / (1 + np.exp(-x))


def nnforward(a, b, x):
    z = []
    for i in range(len(b) - 1):
        z.append(sigm(a[i][0] + a[i][1] * x))
    f = b[0]
    for i in range(len(b) - 1):
        f += b[i+1] * z[i]
    return [z, f]


def nnback(a_old, b_old, x, y, rate):
    pred = nnforward(a_old, b_old, x)
    z = np.array(pred[0])
    z1 = np.r_[1, z]
    f = pred[1]
    b_new = b_old - rate * (-2) * (y - f) * z1
    a_new = a_old - rate * (-2) * (y - f) * np.c_[b_old[1:], b_old[1:]] * np.c_[z * (1 - z), z * (1 - z) * x]
    return [a_new, b_new]


x = np.arange(-1, 1, 0.1)
# ysq = x**2
ysin = np.sin(3*x)

# asq = linreg(x, ysq)
# asin = linreg(x, ysin)

a = np.random.rand(3, 2)
b = np.random.rand(4)
c = [a, b]

plt.subplot(2, 1, 1)
plt.plot(x, ysin, 'o')
for j in range(5):
    for i in range(1000):
        for k in range(len(x)):
            c = nnback(c[0], c[1], x[k], ysin[k], 0.1)
    plt.plot(x, nnforward(c[0], c[1], x)[1], label=j)


plt.legend()
plt.show()
