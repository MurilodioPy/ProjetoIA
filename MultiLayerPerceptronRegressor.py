import pandas as pd
import numpy as np
import math
from MultiLayerPerceptronClassifica import Sigmoid
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class MeanSquaredError:
    def __init__(self):
        pass

    def loss(self, y, p):
        """
        Calcula o Erro Quadrático Médio (MSE).

        Parâmetros:
        - y: Valores reais (targets).
        - p: Valores previstos pelo modelo.

        Retorna:
        - O valor do MSE.
        """
        return np.mean((y - p) ** 2)

    def gradient(self, y, p):
        """
        Calcula o gradiente do MSE com relação às previsões.

        Parâmetros:
        - y: Valores reais (targets).
        - p: Valores previstos pelo modelo.

        Retorna:
        - O gradiente do MSE com relação a p.
        """
        return -2 * (y - p) / len(y)


class MultilayerPerceptronRegressor():
    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.loss = lambda y, p: np.mean((y - p) ** 2)  # MSE

    def _initialize_weights(self, X):
        n_samples, n_features = X.shape
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))
        self.V = np.random.uniform(-limit, limit, (self.n_hidden, 1))  # Uma saída
        self.v0 = np.zeros((1, 1))

    def fit(self, X, y):
        self._initialize_weights(X)
        for _ in range(self.n_iterations):
            hidden_input = X.dot(self.W) + self.w0
            hidden_output = self.hidden_activation(hidden_input)
            output_layer_input = hidden_output.dot(self.V) + self.v0
            y_pred = output_layer_input.flatten()

            # Gradiente
            grad_v = hidden_output.T.dot(2 * (y_pred - y).reshape(-1, 1)) / len(X)
            grad_v0 = np.sum(2 * (y_pred - y).reshape(-1, 1)) / len(X)
            grad_w = X.T.dot(((2 * (y_pred - y).reshape(-1, 1)) * self.hidden_activation.gradient(hidden_input)).reshape(-1, self.n_hidden)) / len(X)
            grad_w0 = np.sum(2 * (y_pred - y).reshape(-1, 1) * self.hidden_activation.gradient(hidden_input), axis=0) / len(X)

            # Atualização dos pesos
            self.V -= self.learning_rate * grad_v
            self.v0 -= self.learning_rate * grad_v0
            self.W -= self.learning_rate * grad_w
            self.w0 -= self.learning_rate * grad_w0

    def predict(self, X):
        hidden_input = X.dot(self.W) + self.w0
        hidden_output = self.hidden_activation(hidden_input)
        output_layer_input = hidden_output.dot(self.V) + self.v0
        return output_layer_input.flatten()  # Garantir que y_pred seja 1D



# Carregar o CSV
data = pd.read_csv('./dataset/vale.csv', delimiter=',')  # Altere o caminho para o seu arquivo CSV

data = pd.DataFrame(data)

# Converter as datas para variáveis numéricas (ex: número de dias desde o primeiro registro)
data['DATA'] = pd.to_datetime(data['DATA'], format='%d/%m/%Y')
data['DIA'] = (data['DATA'] - data['DATA'].min()).dt.days
data['FECHAMENTO'] = data['FECHAMENTO'].str.replace(',', '.').astype(float)

data = data.dropna()

# Prever a coluna "FECHAMENTO" com base no "DIA"
X = data[['DIA']].values
y = data['FECHAMENTO'].values


# Dividir os dados em conjunto de treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Criar e treinar o modelo
mlp = MultilayerPerceptronRegressor(n_hidden=30, n_iterations=3000, learning_rate=0.01)
mlp.fit(X_train, y_train)

# Fazer previsões
y_pred = mlp.predict(X_val)

# Calcular o erro médio absoluto
mae = np.mean(np.abs(y_val - y_pred))
print("Erro Médio Absoluto (MAE):", mae)


# Criando o gráfico
plt.figure(figsize=(10, 6))

# Plotando os valores reais
plt.plot(y_val, label='Valores Reais', color='blue', marker='o', linestyle='None')

# Plotando os valores previstos
plt.plot(y_pred, label='Valores Previstos', color='red', marker='x', linestyle='None')

# Adicionando título e rótulos aos eixos
plt.title('Valores Reais vs Valores Previstos')
plt.xlabel('Amostras')
plt.ylabel('Fechamento')

# Adicionando legenda
plt.legend()

# Exibindo o gráfico
plt.show()

