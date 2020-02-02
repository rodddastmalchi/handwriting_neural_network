import numpy as np
from mlxtend.data import loadlocal_mnist


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


X, y = loadlocal_mnist(
        images_path='train-images.idx3-ubyte',
        labels_path='train-labels.idx1-ubyte')

# print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
# print('\n1st row', X[0])
# print(len(X[0]))
# print('\nfirst label', y[0])

np.random.seed(1)
weights = np.random.rand(784)
result = []
input = []
for i in X[0]:
    if i != 0:
        input.append(1)
    else:
        input.append(0)

input = np.asarray(input)

for i in range(1):
    for j in range(len(input)):
        print(np.dot(input[j], weights[j]))
        result.append(sigmoid(np.dot(input[j], 100*weights[j])))

print(f'Result after training {result}')