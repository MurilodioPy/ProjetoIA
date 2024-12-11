import numpy as np

class Madaline:
    def __init__(self, tm_entrada, num_neuro):
        # Inicializa os pesos (num_neurons, input_size + 1, pois há o bias)
        self.weights = np.random.randn(num_neuro, tm_entrada + 1)  # Inclui o bias
        self.num_neuro = num_neuro

    def activation(self, x):
        # Função degrau bipolar
        n = np.where(x >= 0, 1, -1)
        return n

    # Calcula a precisão
    def calcular_acuracia(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def predict(self, X):
        # Adiciona o bias (coluna de 1s)
        X_bias = np.c_[np.ones(X.shape[0]), X]  # Adiciona a coluna de bias com 1s

        # calcula a saída para cada neurônio
        outputs = self.activation(np.dot(X_bias, self.weights.T))  # Multiplica X_bias por weights transposta
        return np.sign(np.sum(outputs, axis=1))  # Soma os outputs para cada amostra

    def train(self, X, y, epoca=100, learning_rate=0.01):
        # Certifique-se de que X seja um numpy.ndarray
        X = np.array(X)

        # Adiciona o bias às entradas ou seja uma coluna com valores 1
        X_bias = np.c_[np.ones(X.shape[0]), X]
        total_error = 0  # Inicializa o erro total da época

        for epoca in range(epoca):

            for i, x in enumerate(X_bias):
                # i = indice
                # x = bias e entradas
                # Calcula a saída da rede
                output = self.activation(np.dot(self.weights, x))

                # np.sum(output)
                # Essa soma indica a "força" do voto da maioria dos neurônios.
                # Um valor positivo sugere que a maioria dos neurônios está votando 1

                # np.sign(valor)
                # 1, se o valor for positivo (value>0),
                # 0, se o valor for zero (value = 0),
                # -1, se o valor for negativo (value < 0)
                y_pred = np.sign(np.sum(output))

                # Se a predição estiver errada, atualiza os pesos
                if y_pred != y[i]:
                    total_error += abs(y[i] - y_pred)  # Soma o erro absoluto
                    for j in range(self.num_neuro):
                        if output[j] != y[i]:
                            # Ajusta apenas os neurônios que erraram
                            self.weights[j] += learning_rate * (y[i] - output[j]) * x
            # Para o treinamento se não houver erros
            if total_error == 0:
                print("Treinamento concluído sem erros.")
                break

            # Exibe o erro total por época
        print(f"Época {epoca + 1}, Erro Total: {total_error}")

