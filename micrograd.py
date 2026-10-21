import math
import numpy as np
import matplotlib.pyplot as plt

def parabola(a, b, c, x):
    return a*x**2 + b*x + c

xs = np.arange(-5, 5, 0.25)
ys = parabola(3, 4, 5, xs)

for x,y in zip(xs,ys):
    print(f"X: {x}, Y: {y}")

plt.plot(xs, ys)