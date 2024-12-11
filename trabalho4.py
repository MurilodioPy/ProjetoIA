import numpy as np
from MultiLayerPerceptron import MultiLayerperceptron
import matplotlib.pyplot as plt

# Geração de dados
def generate_data(func, n_samples=100, noise=0.1):
    X = np.random.uniform(-2, 2, (n_samples, 1))  # Valores de entrada no intervalo [-2, 2]
    y = func(X) + noise * np.random.randn(n_samples, 1)  # Função com ruído
    return X, y

# Função "estranha" que queremos aproximar
def strange_function(x):
    return np.sin(5 * x) + np.cos(3 * x) + x**2

# Dados de treino e teste
X_train, y_train = generate_data(strange_function, n_samples=200)
X_test = np.linspace(-2, 2, 100).reshape(-1, 1)  # Testar em pontos uniformemente distribuídos
y_test = strange_function(X_test)

# Configuração e treinamento do modelo
mlp = MultiLayerperceptron(input_size=1, hidden_size=10, output_size=1, learning_rate=0.01)
mlp.train(X_train, y_train, epochs=5000)

# Previsões
y_pred = mlp.predict(X_test)

# Plot dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color="blue", label="Dados de Treinamento", alpha=0.5)
plt.plot(X_test, y_test, color="green", label="Função Verdadeira")
plt.plot(X_test, y_pred, color="red", linestyle="--", label="Aproximação MLP")
plt.legend()
plt.title("Aproximação de Função com Multilayer Perceptron")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()