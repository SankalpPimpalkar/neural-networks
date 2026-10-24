from value import Value
from dd import draw
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum([wi*xi for wi, xi in zip(self.w, x)], self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params


xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]
n = MLP(3, [4,4,1])

ypred = [n(x) for x in xs]
loss = sum([(yc - yd)**2 for yc, yd in zip(ypred, ys)])

step_sz = 0.01
print("Total Parameters", len(n.parameters()))

# print(n.layers[0].neurons[0].w[0].data)
ypred = [n(x) for x in xs]
loss = sum([(yc - yd)**2 for yc, yd in zip(ypred, ys)])
print("Initial Loss: ", loss)


for i in range(100):
    for p in n.parameters():
        p.data += -step_sz * p.grad

    for p in n.parameters():
        p.grad = 0.0

    # print(n.layers[0].neurons[0].w[0].data)
    ypred = [n(x) for x in xs]
    loss = sum([(yc - yd)**2 for yc, yd in zip(ypred, ys)])
print("Final Loss: ", loss)

draw(loss)