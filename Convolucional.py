import numpy as np
from PIL import Image

# Função para carregar e preprocessar imagens
def load_and_preprocess_images(image_paths, size=(32, 32)):
    images = []
    labels = []
    for label, path in enumerate(image_paths):
        image = Image.open(path).convert('L')  # Converte para escala de cinza
        image = image.resize(size)  # Redimensiona
        image_array = np.array(image) / 255.0  # Normaliza para [0, 1]
        images.append(image_array)
        labels.append(label)
    return np.array(images), np.array(labels)

# Funções para a CNN
def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    output = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            region = image[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)
    return output

def max_pooling(image, pool_size=(2, 2)):
    h, w = image.shape
    ph, pw = pool_size
    output = np.zeros((h // ph, w // pw))
    for i in range(0, h, ph):
        for j in range(0, w, pw):
            region = image[i:i + ph, j:j + pw]
            output[i // ph, j // pw] = np.max(region)
    return output

def flatten(image):
    return image.flatten()

def dense_forward(inputs, weights, biases):
    return np.dot(inputs, weights) + biases

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# Cálculo de perda e precisão
def cross_entropy_loss(predictions, labels):
    m = predictions.shape[0]
    log_likelihood = -np.log(predictions[range(m), labels])
    return np.sum(log_likelihood) / m

def accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == labels)

# Determinar tamanho do vetor flatten
def calculate_flattened_size(image_size, filter_size, pooling_size, num_filters):
    conv_output_size = image_size - filter_size + 1
    pooled_output_size = conv_output_size // pooling_size
    return (pooled_output_size ** 2) * num_filters

# Aplicar múltiplos filtros
def apply_filters(image, filters):
    filtered_images = []
    for kernel in filters:
        filtered_image = convolve(image, kernel)
        filtered_images.append(filtered_image)
    return np.array(filtered_images)

# Treinamento da CNN
def train_cnn(images, labels, filters, epochs=10, learning_rate=0.01):
    np.random.seed(0)
    image_size = images[0].shape[0]
    filter_size = filters[0].shape[0]
    pooling_size = 2
    num_filters = len(filters)

    n_features = calculate_flattened_size(image_size, filter_size, pooling_size, num_filters)
    n_classes = len(np.unique(labels))
    weights = np.random.randn(n_features, n_classes) * 0.01
    biases = np.zeros((n_classes,))

    for epoch in range(epochs):
        total_loss = 0
        for i, image in enumerate(images):
            # Forward pass
            filtered_outputs = apply_filters(image, filters)
            pooled_outputs = [max_pooling(filtered) for filtered in filtered_outputs]
            flattened_output = np.concatenate([flatten(pooled) for pooled in pooled_outputs])
            logits = dense_forward(flattened_output, weights, biases)
            probs = softmax(logits)

            # Loss and backpropagation
            loss = -np.log(probs[labels[i]])
            total_loss += loss
            d_logits = probs
            d_logits[labels[i]] -= 1
            d_weights = np.outer(flattened_output, d_logits)
            d_biases = d_logits

            weights -= learning_rate * d_weights
            biases -= learning_rate * d_biases

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(images)}")

    return weights, biases

# Previsão
def predict(image, filters, weights, biases):
    filtered_outputs = apply_filters(image, filters)
    pooled_outputs = [max_pooling(filtered) for filtered in filtered_outputs]
    flattened_output = np.concatenate([flatten(pooled) for pooled in pooled_outputs])
    logits = dense_forward(flattened_output, weights, biases)
    return softmax(logits)

# Caminhos das imagens
image_paths = [
    "./dataset/superman1.jpeg",
    "./dataset/superman2.jpeg",  
    "./dataset/superman3.jpeg",
    "./dataset/batman1.jpeg",
    "./dataset/batman2.jpeg",
    "./dataset/batman3.jpeg"
]

test_image = [
    "./dataset/superman.jpeg",
    "./dataset/batman.jpeg"
]

# Filtros
filters = [
    np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),  # Bordas horizontais
    np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),  # Bordas verticais
    np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])   # Nitidez
]

# Carregar e preprocessar as imagens
images, labels = load_and_preprocess_images(image_paths)

images_teste, labels_teste = load_and_preprocess_images(test_image)

# Treinar a rede neural
weights, biases = train_cnn(images, labels, filters, epochs=300, learning_rate=0.01)

# 0 SuperMan
# 1 Batman
test_image = images_teste[1] 

predictions = predict(test_image, filters, weights, biases)
print(f"Predicted probabilities: {predictions}")
print(f"Predicted class: {np.argmax(predictions)}")
