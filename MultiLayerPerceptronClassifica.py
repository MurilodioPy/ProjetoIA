#######################################################################
import numpy as np
import math


#######################################################################

def normalize(X, axis=-1, order=2):
    """ Normaliza o dataset X utilizando a norma L2 """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))  # Calcula a norma L2 para cada linha
    l2[l2 == 0] = 1  # Evita divisão por zero
    return X / np.expand_dims(l2, axis)  # Retorna o dataset normalizado

########################################################################

def to_categorical(x, n_col=None):
    """ Converte valores nominais (classes) para codificação one-hot """
    if not n_col:  # Determina o número de colunas (classes)
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))  # Cria uma matriz de zeros
    one_hot[np.arange(x.shape[0]), x] = 1  # Define 1 para a classe correspondente
    return one_hot

########################################################################

def accuracy_score(y_true, y_pred):
    """ Calcula a acurácia como a proporção de previsões corretas """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

########################################################################

class CrossEntropy():
    def __init__(self): pass

    def loss(self, y, p):
        """
        Calcula a função de perda da entropia cruzada.

        Parâmetros:
        - y: Valores reais (em codificação one-hot).
        - p: Probabilidades previstas pelo modelo.

        Retorna:
        - Valor da perda para cada exemplo no dataset.
        """
        p = np.clip(p, 1e-15, 1 - 1e-15)  # Evita divisão por zero
        return - y * np.log(p) - (1 - y) * np.log(1 - p)  # Fórmula da entropia cruzada

    def acc(self, y, p):
        """
        Calcula a acurácia do modelo.

        Parâmetros:
        - y: Valores reais (em codificação one-hot).
        - p: Probabilidades previstas pelo modelo.

        Retorna:
        - Acurácia como proporção de previsões corretas.
        """
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        """
        Calcula o gradiente da entropia cruzada em relação às probabilidades previstas.

        Parâmetros:
        - y: Valores reais (em codificação one-hot).
        - p: Probabilidades previstas pelo modelo.

        Retorna:
        - Gradiente da perda com relação às probabilidades previstas.
        """
        p = np.clip(p, 1e-15, 1 - 1e-15)  # Evita divisão por zero
        return - (y / p) + (1 - y) / (1 - p)  # Fórmula do gradiente

#######################################################################

class Sigmoid():
    # Função de ativação sigmoid
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))  # S(x) = 1 / (1 + exp(-x))

    def gradient(self, x):
        # Derivada da função sigmoid
        return self.__call__(x) * (1 - self.__call__(x))  # S'(x) = S(x) * (1 - S(x))

#########################################################################

class Softmax():
    def __call__(self, x):
        # Função de ativação softmax
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Estabiliza numericamente
        return e_x / np.sum(e_x, axis=-1, keepdims=True)  # Normaliza os valores no intervalo (0, 1)

    def gradient(self, x):
        # Gradiente simplificado da softmax
        p = self.__call__(x)
        return p * (1 - p)

########################################################################

class MultilayerPerceptron():
    """
    Implementação de um Perceptron Multicamadas (MLP).
    """
    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01):
        # Configuração inicial
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()  # Ativação da camada oculta
        self.output_activation = Softmax()  # Ativação da camada de saída
        self.loss = CrossEntropy()          # Função de perda

    def _initialize_weights(self, X, y):
        """
        Inicializa os pesos e biases do modelo.
        """
        n_samples, n_features = X.shape
        _, n_outputs = y.shape

        # Pesos da camada oculta
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))

        # Pesos da camada de saída
        limit = 1 / math.sqrt(self.n_hidden)
        self.V = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs))
        self.v0 = np.zeros((1, n_outputs))

    def fit(self, X, y):
        """
        Treina o modelo no conjunto de dados.
        """
        self._initialize_weights(X, y)

        for i in range(self.n_iterations):
            # Forward pass
            hidden_input = X.dot(self.W) + self.w0
            hidden_output = self.hidden_activation(hidden_input)
            output_layer_input = hidden_output.dot(self.V) + self.v0
            y_pred = self.output_activation(output_layer_input)

            # Backward pass
            grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            grad_v = hidden_output.T.dot(grad_wrt_out_l_input)
            grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)

            grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_activation.gradient(hidden_input)
            grad_w = X.T.dot(grad_wrt_hidden_l_input)
            grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

            # Atualização dos pesos
            self.V -= self.learning_rate * grad_v
            self.v0 -= self.learning_rate * grad_v0

            self.W -= self.learning_rate * grad_w
            self.w0 -= self.learning_rate * grad_w0

    def predict(self, X):
        """
        Realiza previsões usando o modelo treinado.
        """
        hidden_input = X.dot(self.W) + self.w0
        hidden_output = self.hidden_activation(hidden_input)
        output_layer_input = hidden_output.dot(self.V) + self.v0
        y_pred = self.output_activation(output_layer_input)
        return y_pred

