import matplotlib.pyplot as plt
import numpy as np
from math import exp
from random import gauss

a, b, c = 1.0, 2.0, 3.0
N = 100
sigma = 1.0
x = [i / N for i in range(N)]
y = [exp(a)*x_*x_ + b*x_ + c for x_ in x]
y2 = [y_ + gauss(0, sigma) for y_ in y]

plt.plot(x, y)
plt.scatter(x, y2, color='r')
plt.show()

np.savetxt('data_x.txt', x, fmt='%s')
np.savetxt('data_y.txt', y2, fmt='%s')
