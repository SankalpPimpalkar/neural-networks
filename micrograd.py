import numpy as np
import matplotlib.pyplot as plt
from dd import dd
from value import Value
from equations import parabola

h = 0.000001
x = 3

# a = Value(2, label='a')
# a.grad = -2 * -3

# b = Value(-3, label='b')
# b.grad = -2 * 2

# e = a * b
# e.label = "e"
# e.grad = -2

# c = Value(10, label='c') 
# c.grad = -2 

# d = e + c
# d.label = "d"
# d.grad = -2

# f = Value(-2, label='f')
# f.grad = 4

# L = d * f
# L.label = "L"
# L.grad = 1

x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

b = Value(6.881, label='b')

x1w1 = x1 * w1
x1w1.label = "x1w1"

x2w2 = x2 * w2
x2w2.label = "x2w2"

x1w1_add_x2w2 = x1w1 + x2w2
x1w1_add_x2w2.label = "x1w1 + x2w2"

n = x1w1_add_x2w2 + b
n.label = "n"

e = (2*n).exp()

out = (e - 1) / (e + 1)
out.label = "O"

out.backward()

dd(out).render("draw")