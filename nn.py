import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

sizes = [2, 3, 3, 1]
n = len(sizes)

x = np.array([0.5, 0.2])
weights = []
biases = []

for i in range(n-1):
    weights.append(np.random.randn(sizes[i], sizes[i+1]))
    biases.append(np.zeros(sizes[i+1]))

output = x
for w,b in zip(weights, biases):
    output = sigmoid(output @ w + b)

print("Output: ", output)