import numpy as np

# Função de ativação step
# retorna 1 se o valor de entrada z for maior ou igual a 0, e 0 caso contrário.
def step_function(z):
    return 1 if z >= 0 else 0

# Implementação do Perceptron
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, ciclos=10):
        self.weights = np.random.uniform(-0.5, 0.5, input_size)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate
        self.ciclos = ciclos

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return step_function(z)

    def train(self, entradas, saidas):
        for _ in range(self.ciclos):
            for i in range(len(entradas)):
                print(self.weights)
                prediction = self.predict(entradas[i])
                self.weights += self.learning_rate * (saidas[i] - prediction) * entradas[i]
                self.bias += self.learning_rate * (saidas[i] - prediction)



























# # Dados de treino (AND lógico)
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0, 0, 0, 1])
#
# # Treinando o Perceptron
# p = Perceptron(input_size=2)
# p.train(X, y)
#
# # Testando o Perceptron
# for x in X:
#     print(f"{x} -> {p.predict(x)}")
