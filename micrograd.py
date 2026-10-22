import numpy as np
import matplotlib.pyplot as plt
from dd import dd
from value import Value
from equations import parabola

xs = np.arange(-5, 5, 0.25)
ys = parabola(xs, 3, 4, 5)

for x, y in zip(xs, ys):
    # print(f"X: {x}, Y: {y}")
    pass


h = 0.000001
x = 3
# print((parabola(x + h) - parabola(x)) / h)
# plt.plot(xs, ys)

a = Value(3, label='a')
b = Value(4, label='b')
c = Value(5, label='c')

d = a * b
d.label = 'd'

e = d + c
e.label = 'e'
e.grad = 1

dd(e).render("draw")