import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 데이터 설정
# X: 독립 변수, 2개의 특징(X1, X2)로 구성된 10개의 샘플
# y: 종속 변수(Class), 이진 분류 (0 또는 1)
X = np.array([[0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649],
              [0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595]])
y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

# Logistic Regression 구현
# 학습률(lr)과 반복 횟수(epochs)를 설정할 수 있는 단순한 Logistic Regressor 클래스
class LogisticRegressor:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr  # 학습률
        self.epochs = epochs  # 반복 횟수

    # Sigmoid 함수: z 값을 0과 1 사이의 확률로 변환
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 학습 함수: 주어진 데이터를 사용해 가중치와 바이어스를 업데이트
    def train(self, X, y):
        self.weights = np.zeros(X.shape[0])  # 가중치 초기화 (0으로 설정)
        self.bias = 0  # 바이어스 초기화

        for _ in range(self.epochs):  # epochs만큼 반복
            # 선형 모델 계산 (z = w * X + b)
            linear_model = np.dot(self.weights, X) + self.bias
            # Sigmoid를 통해 예측 확률 계산
            y_predicted = self.sigmoid(linear_model)
            # 예측값과 실제값의 차이 계산 (오차)
            error = y_predicted - y

            # Gradient Descent를 사용해 가중치와 바이어스 업데이트
            dw = np.dot(X, error) / X.shape[1]  # 가중치의 기울기
            db = np.sum(error) / X.shape[1]  # 바이어스의 기울기

            self.weights -= self.lr * dw  # 가중치 업데이트
            self.bias -= self.lr * db  # 바이어스 업데이트

    # 예측 함수: 학습된 모델로 X 데이터를 분류
    def predict(self, X):
        linear_model = np.dot(self.weights, X) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]  # 임계값 0.5로 분류

# MLP 구현
# 1개의 은닉층을 가지는 Multi-Layer Perceptron 클래스
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, epochs=1000):
        self.lr = lr  # 학습률
        self.epochs = epochs  # 반복 횟수
        self.weights1 = np.random.rand(hidden_size, input_size)  # 은닉층 가중치 초기화
        self.bias1 = np.random.rand(hidden_size, 1)  # 은닉층 바이어스 초기화
        self.weights2 = np.random.rand(output_size, hidden_size)  # 출력층 가중치 초기화
        self.bias2 = np.random.rand(output_size, 1)  # 출력층 바이어스 초기화

    # Sigmoid 함수
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Sigmoid 함수의 도함수: Backpropagation에서 사용
    def sigmoid_derivative(self, z):
        return z * (1 - z)

    # 학습 함수: Forward와 Backward Propagation을 통해 가중치와 바이어스를 최적화
    def train(self, X, y):
        y = y.reshape(1, -1)  # y의 형태를 (1, n)으로 변환
        for _ in range(self.epochs):
            # Forward Propagation
            z1 = np.dot(self.weights1, X) + self.bias1  # 은닉층 선형 결합
            a1 = self.sigmoid(z1)  # 은닉층 활성화 함수 적용
            z2 = np.dot(self.weights2, a1) + self.bias2  # 출력층 선형 결합
            a2 = self.sigmoid(z2)  # 출력층 활성화 함수 적용

            # Backward Propagation
            error = a2 - y  # 출력층 오차
            dz2 = error * self.sigmoid_derivative(a2)  # 출력층의 기울기
            dw2 = np.dot(dz2, a1.T)  # 출력층 가중치 기울기
            db2 = np.sum(dz2, axis=1, keepdims=True)  # 출력층 바이어스 기울기

            dz1 = np.dot(self.weights2.T, dz2) * self.sigmoid_derivative(a1)  # 은닉층의 기울기
            dw1 = np.dot(dz1, X.T)  # 은닉층 가중치 기울기
            db1 = np.sum(dz1, axis=1, keepdims=True)  # 은닉층 바이어스 기울기

            # Gradient Descent
            self.weights2 -= self.lr * dw2  # 출력층 가중치 업데이트
            self.bias2 -= self.lr * db2  # 출력층 바이어스 업데이트
            self.weights1 -= self.lr * dw1  # 은닉층 가중치 업데이트
            self.bias1 -= self.lr * db1  # 은닉층 바이어스 업데이트

    # 예측 함수: 학습된 MLP로 X 데이터를 분류
    def predict(self, X):
        z1 = np.dot(self.weights1, X) + self.bias1
        a1 = self.sigmoid(z1)
        z2 = np.dot(self.weights2, a1) + self.bias2
        a2 = self.sigmoid(z2)
        return [1 if i > 0.5 else 0 for i in a2.flatten()]

# Logistic Regressor 학습 및 평가
print("Logistic Regressor")
lr_model = LogisticRegressor(lr=0.5, epochs=10000)
lr_model.train(X, y)
y_pred_lr = lr_model.predict(X)
print(confusion_matrix(y, y_pred_lr))
print(classification_report(y, y_pred_lr))
print('w: ', lr_model.weights, 'b: ', lr_model.bias)

# MLP 학습 및 평가
print("\nMLP")
mlp_model = MLP(input_size=2, hidden_size=5, output_size=1, lr=0.5, epochs=1000)
mlp_model.train(X, y)
y_pred_mlp = mlp_model.predict(X)
print(confusion_matrix(y, y_pred_mlp))
print(classification_report(y, y_pred_mlp))

# Logistic Regression의 결정 경계 계산 및 시각화
w = lr_model.weights
b = lr_model.bias
x1_values = np.linspace(0, 1, 100)
x2_values = -(w[0] * x1_values + b) / w[1]

# 데이터 분리
X1_class0 = [x1 for x1, c in zip(X[0], y) if c == 0]
X2_class0 = [x2 for x2, c in zip(X[1], y) if c == 0]
X1_class1 = [x1 for x1, c in zip(X[0], y) if c == 1]
X2_class1 = [x2 for x2, c in zip(X[1], y) if c == 1]

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.scatter(X1_class0, X2_class0, color='red', label='Class 0', edgecolor='k', s=100, alpha=0.8)
plt.scatter(X1_class1, X2_class1, color='blue', label='Class 1', edgecolor='k', s=100, alpha=0.8)
plt.plot(x1_values, x2_values, color='green', label='Decision Boundary', linewidth=2)
plt.title('Logistic Regression Decision Boundary', fontsize=14)
plt.xlabel('X1', fontsize=12)
plt.ylabel('X2', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
