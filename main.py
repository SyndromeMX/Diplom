import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        """Прямое распространение"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        """Обратное распространение ошибки"""
        loss = output - y
        d_output = loss * self.sigmoid_derivative(output)

        hidden_error = d_output.dot(self.W2.T)
        d_hidden = hidden_error * self.sigmoid_derivative(self.a1)

        # Обновление весов
        self.W2 -= self.a1.T.dot(d_output) * self.learning_rate
        self.b2 -= np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.W1 -= X.T.dot(d_hidden) * self.learning_rate
        self.b1 -= np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Эпоха {epoch}, Ошибка: {loss}")


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])  


mlp = MultiLayerPerceptron(input_size=2, hidden_size=4, output_size=1)
mlp.train(X, y, epochs=10000)


print("Результаты после обучения:")
for i in X:
    print(f"{i} -> {mlp.forward(i)}")
