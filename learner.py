import numpy as np


class NN:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, batch_size, eta, test_data=[]):
        if test_data:
            num_tests = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            # split up training data into batches, test each batch independently
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]

            # update weights for each batch
            for batch in mini_batches:
                self.update_batch(batch, eta)

            if test_data:
                print(f'Epoch {j}: {self.evaluate(test_data)} / {num_tests}')
            else:
                print(f'Epoch {j} completed')

    def update_batch(self, batch, eta):
        b_gradient = [np.zeros(b.shape) for b in self.biases]
        w_gradient = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_b_gradient, delta_w_gradient = self.backprop(x, y)
            b_gradient = [nb+dnb for nb, dnb in zip(b_gradient, delta_b_gradient)]
            w_gradient = [nb + dnb for nb, dnb in zip(w_gradient, delta_w_gradient)]
        self.weights = [w - (eta/len(batch))*nw for w, nw in zip(self.weights, w_gradient)]
        self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, b_gradient)]

    def backprop(self, x, y):
        """Return a tuple ``(b_gradient, w_gradient)`` representing the
        gradient for the cost function C_x.  ``b_gradient`` and
        ``w_gradient`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        b_gradient = [np.zeros(b.shape) for b in self.biases]
        w_gradient = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        b_gradient[-1] = delta
        w_gradient[-1] = np.dot(delta, activations[-2].T)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            b_gradient[-l] = delta
            w_gradient[-l] = np.dot(delta, activations[-l-1].T)
        return (b_gradient, w_gradient)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        # count =0
        # for x in test_results:
        #     if x[0] == x[1]:
        #         count += 1
        # return count

        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


