import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 데이터셋 준비
transform_test = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

class AlexNet(nn.Module):
    def __init__(self, num_cls=1000):
        super(AlexNet, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.skip1 = self.skip_connection(96, 384)

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)

        self.skip2 = self.skip_connection(384, 256)

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_cls),
        )

    def skip_connection(self, in_dim, out_dim):
        sc = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2),
            nn.BatchNorm2d(out_dim)
        )
        return sc
    
    def feature(self, x):
        x = self.conv1(x)

        skip1_out = self.skip1(x)
        x = self.conv2(x)
        x = self.relu(self.conv3(x) + skip1_out)

        skip2_out = self.skip2(x)
        x = self.conv4(x)
        x = self.relu(self.conv5(x) + skip2_out)

        return x

    def forward(self, x):
        x = self.feature(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AlexNet(num_cls=10).to(device)  
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# 3. 임베딩 추출
embeddings = []
labels = []

with torch.no_grad():
    for images, targets in test_loader:
        # 모델에서 임베딩 추출
        features = model.feature(images.to(device))
        features = features.view(features.size(0), -1)  # Flatten
        embeddings.append(features.cpu().numpy())
        labels.append(targets.cpu().numpy())

# numpy 배열로 변환
embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)

# 2. 평균 중심화
mean_data = np.mean(embeddings, axis=0)
centered_data = embeddings - mean_data

# 3. 공분산 행렬 계산
cov_matrix = np.cov(centered_data.T)  # (특성 x 특성) 크기의 공분산 행렬

# 4. 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 5. 고유값 기준으로 내림차순 정렬
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# 6. 상위 2개의 주성분 선택
top_2_eigenvectors = sorted_eigenvectors[:, :2]

# 7. 데이터 투영 (Embedding)
embedded_data = np.dot(centered_data, top_2_eigenvectors)

# 8. 시각화
class_names = test_dataset.classes
colors = plt.cm.get_cmap('tab10', len(class_names))

plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    idxs = labels == i
    plt.scatter(embedded_data[idxs, 0], embedded_data[idxs, 1], label=class_names[i], alpha=0.6, c=[colors(i)])

plt.legend()
plt.title("2D PCA of Embedding Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()