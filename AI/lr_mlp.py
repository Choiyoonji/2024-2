import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 데이터 설정
X = np.array([[0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649],
              [0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595]])
y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

# Logistic Regression 구현
class LogisticRegressor:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        self.weights = np.zeros(X.shape[0])
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(self.weights, X) + self.bias
            y_predicted = self.sigmoid(linear_model)
            error = y_predicted - y

            # Gradient Descent
            dw = np.dot(X, error) / X.shape[1]
            db = np.sum(error) / X.shape[1]

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(self.weights, X) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]


# MLP 구현
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights1 = np.random.rand(hidden_size, input_size)
        self.bias1 = np.random.rand(hidden_size, 1)
        self.weights2 = np.random.rand(output_size, hidden_size)
        self.bias2 = np.random.rand(output_size, 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def train(self, X, y):
        y = y.reshape(1, -1)
        for _ in range(self.epochs):
            # Forward pass
            z1 = np.dot(self.weights1, X) + self.bias1
            a1 = self.sigmoid(z1)
            z2 = np.dot(self.weights2, a1) + self.bias2
            a2 = self.sigmoid(z2)

            # Backward pass
            error = a2 - y
            dz2 = error * self.sigmoid_derivative(a2)
            dw2 = np.dot(dz2, a1.T)
            db2 = np.sum(dz2, axis=1, keepdims=True)

            dz1 = np.dot(self.weights2.T, dz2) * self.sigmoid_derivative(a1)
            dw1 = np.dot(dz1, X.T)
            db1 = np.sum(dz1, axis=1, keepdims=True)

            # Gradient Descent
            self.weights2 -= self.lr * dw2
            self.bias2 -= self.lr * db2
            self.weights1 -= self.lr * dw1
            self.bias1 -= self.lr * db1

    def predict(self, X):
        z1 = np.dot(self.weights1, X) + self.bias1
        a1 = self.sigmoid(z1)
        z2 = np.dot(self.weights2, a1) + self.bias2
        a2 = self.sigmoid(z2)
        return [1 if i > 0.5 else 0 for i in a2.flatten()]


# 모델 훈련 및 평가
print("Logistic Regressor")
lr_model = LogisticRegressor(lr=0.1, epochs=1000)
lr_model.train(X, y)
y_pred_lr = lr_model.predict(X)
print(confusion_matrix(y, y_pred_lr))
print(classification_report(y, y_pred_lr))

print("\nMLP")
mlp_model = MLP(input_size=2, hidden_size=5, output_size=1, lr=0.1, epochs=1000)
mlp_model.train(X, y)
y_pred_mlp = mlp_model.predict(X)
print(confusion_matrix(y, y_pred_mlp))
print(classification_report(y, y_pred_mlp))
