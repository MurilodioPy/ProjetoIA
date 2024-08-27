import numpy as np

def step_function(z):
    return 1 if z >= 0 else 0

class Hebb:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0.0

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return step_function(z)

    def train(self, entradas, saidas):
        for cont1 in range(2):
            deltaW = entradas[cont1] * saidas[cont1]
            deltaB = saidas[cont1]
            self.weights += deltaW
            self.bias += deltaB
