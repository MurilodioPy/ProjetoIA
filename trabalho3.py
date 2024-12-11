import numpy as np
from Madaline import Madaline

# Inicializa o número de entradas e neurônios
input_size = 10  # Número de entradas
num_neuro = 4  # Número de neurônios na camada

# Dados de entrada estáticos para treinamento
X_train = np.array([
    [0.5, -0.2, 0.1, -0.3, 0.8, 0.2, -0.6, 0.4, 0.9, -0.1],
    [-0.7, 0.3, -0.8, 0.6, -0.1, -0.5, 0.3, -0.2, 0.1, 0.4],
    [0.2, 0.8, 0.5, -0.4, 0.7, -0.3, 0.6, 0.1, 0.9, 0.3],
    [-0.9, -0.6, 0.3, -0.7, -0.8, 0.2, 0.4, -0.3, -0.5, -0.1],
    [-0.6, -0.1, 0.6, -0.2, -0.7, 0.4, 0.8, -0.6, -0.2, -0.7],
    [-0.4, -0.5, 0.7, -0.1, -0.3, 0.6, 0.9, -0.5, -0.7, -0.1]
])

# Saídas desejadas estáticas para treinamento
y_train = np.array([1, -1, 1, -1, -1, 1])  # Saídas fixas relacionadas aos dados de entrada

# Dados de entrada estáticos para teste
X_test = np.array([
    [-0.6, -0.1, 0.6, -0.2, -0.7, 0.4, 0.8, -0.6, -0.2, -0.7],
    [-0.4, -0.5, 0.7, -0.1, -0.3, 0.6, 0.9, -0.5, -0.7, -0.1]
])

# Saídas reais do teste (baseadas na mesma lógica do treinamento)
y_test = np.array([-1, 1])

# Instancia e treina o modelo Madaline
madaline = Madaline(tm_entrada=input_size, num_neuro=num_neuro)

print("Iniciando o treinamento...")
madaline.train(X_train, y_train)

print("Treinamento concluído!")

print('\n##  Modelo ##\n')
# Fazer previsões no conjunto de teste
predictions = madaline.predict(X_train)

# Exibir os resultados
print("Saídas previstas:", predictions)
print("Saídas reais:", y_train)

# Calcular a precisão
accuracy = madaline.calcular_acuracia(y_train, predictions)
print(f"Precisão no teste: {accuracy * 100:.2f}%")

print('\n##  Teste ##\n')
# Fazer previsões no conjunto de teste
predictions = madaline.predict(X_test)

# Exibir os resultados
print("Saídas previstas:", predictions)
print("Saídas reais:", y_test)

# Calcular a precisão
accuracy = madaline.calcular_acuracia(y_test, predictions)
print(f"Precisão no teste: {accuracy * 100:.2f}%")
