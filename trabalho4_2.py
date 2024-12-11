from MultiLayerPerceptronClassifica import to_categorical, MultilayerPerceptron, normalize, accuracy_score
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split


##########################################################
# Carrega o dataset Digits
data = datasets.load_digits()
X = normalize(data.data)  # Normaliza os dados
y = data.target  # Classes

# Converte os rótulos para one-hot encoding
y = to_categorical(y)

# Divide o conjunto em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Cria e treina o MLP
clf = MultilayerPerceptron(n_hidden=16, n_iterations=1000, learning_rate=0.01)
clf.fit(X_train, y_train)

# Realiza previsões
y_pred = np.argmax(clf.predict(X_test), axis=1)
y_test = np.argmax(y_test, axis=1)

# Calcula a acurácia
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>")