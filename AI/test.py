import numpy as np

# 데이터
X = np.array([[0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649],
              [0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595]])
y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

# 데이터 전치 및 편향 추가
X = X.T  # (10, 2) 형태로 전치
m, n = X.shape  # m: 샘플 수, n: 특징 수
X = np.hstack([np.ones((m, 1)), X])  # 편향 추가
theta = np.zeros(n + 1)  # 초기 가중치

# 하이퍼파라미터
alpha = 0.5  # 학습률 조정
num_iterations = 5000  # 반복 횟수 증가

# 시그모이드 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 경사 하강법
for _ in range(num_iterations):
    z = np.dot(X, theta)  # 선형 조합
    h = sigmoid(z)  # 예측 확률
    gradient = np.dot(X.T, (h - y)) / m  # 그래디언트
    theta -= alpha * gradient  # 가중치 업데이트

# 학습 완료 후 예측
y_pred = sigmoid(np.dot(X, theta)) >= 0.5
accuracy = np.mean(y_pred == y)

print("최종 가중치:", theta)
print("예측 라벨:", y_pred.astype(int))
print("정확도:", accuracy)
