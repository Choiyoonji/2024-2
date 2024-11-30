import numpy as np
import matplotlib.pyplot as plt

# 데이터 설정
X1 = [0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649]
X2 = [0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595]
Class = [1, 1, 0, 1, 0, 0, 0, 1, 0, 1]

# Logistic Regression 학습 결과 (예시 값)
# 이 값은 Logistic Regression 학습 후 도출된 결과를 사용해야 합니다.
w = np.array([1.86279283, 2.93694608])  # 학습된 가중치
b = -3.2854762222562686  # 학습된 바이어스

# 결정 경계 계산 (w1 * X1 + w2 * X2 + b = 0)
# X2 = -(w1 * X1 + b) / w2
x1_values = np.linspace(0, 1, 100)
x2_values = -(w[0] * x1_values + b) / w[1]

# 데이터 분리
X1_class0 = [x1 for x1, c in zip(X1, Class) if c == 0]
X2_class0 = [x2 for x2, c in zip(X2, Class) if c == 0]
X1_class1 = [x1 for x1, c in zip(X1, Class) if c == 1]
X2_class1 = [x2 for x2, c in zip(X2, Class) if c == 1]

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
