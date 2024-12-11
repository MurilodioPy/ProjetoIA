import numpy as np

class MultiLayerperceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Inicializa pesos e bias para camadas ocultas e de saída
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Pesos e bias para camada escondida
        self.weights_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))

        # Pesos e bias para camada de saída
        self.weights_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward(self, X):
        # Propagação direta
        # Entrada da camada escondida
        self.hidden_input = np.dot(X, self.weights_hidden) + self.bias_hidden

        # Saída da camada escondida
        self.hidden_output = self.sigmoid(self.hidden_input)
        # 
        self.output_input = np.dot(self.hidden_output, self.weights_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)
        print('Saida: ', self.output)
        return self.output

    def backward(self, X, y, output):
        # Cálculo dos gradientes para a camada de saída
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        # Cálculo dos gradientes para a camada escondida
        hidden_error = np.dot(output_delta, self.weights_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Atualização dos pesos e bias da camada de saída
        self.weights_output += np.dot(self.hidden_output.T, output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

        # Atualização dos pesos e bias da camada escondida
        self.weights_hidden += np.dot(X.T, hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):

            output = self.forward(X)

            # Backward pass
            self.backward(X, y, output)

            if epoch % 1000 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

    def predict(self, X):
        # Propagação direta para predizer
        return self.forward(X)


# Exemplo de uso
# if __name__ == "__main__":
#     # Dados simples (porta lógica XOR)
#     X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     y = np.array([[0], [1], [1], [0]])
#
#     # Inicializando o MLP
#     mlp = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
#
#     # Treinando o modelo
#     mlp.train(X, y, epochs=1000)
#
#     # Testando
#     predictions = mlp.predict(X)
#     print("Predictions:")
#     print(predictions)
