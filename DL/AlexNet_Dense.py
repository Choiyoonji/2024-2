import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

class DenseAlexNet(nn.Module):
    def __init__(self, num_cls):
        super(DenseAlexNet, self).__init__()

        # 각 층의 출력 채널 크기
        self.conv1_out_channels = 96
        self.conv2_out_channels = 256
        self.conv3_out_channels = 384
        self.conv4_out_channels = 384
        self.conv5_out_channels = 256

        # 각 컨볼루션 층 정의
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.conv1_out_channels, kernel_size=3, stride=1, padding=1),  # 크기 유지
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 크기 절반으로 감소
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, kernel_size=3, stride=1, padding=1),  # 크기 유지
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 크기 절반으로 감소
        )

        self.conv3 = nn.Conv2d(self.conv1_out_channels + self.conv2_out_channels, self.conv3_out_channels, kernel_size=3, stride=1, padding=1)  # 크기 유지

        self.conv4 = nn.Conv2d(self.conv1_out_channels + self.conv2_out_channels + self.conv3_out_channels, self.conv4_out_channels, kernel_size=3, stride=1, padding=1)  # 크기 유지

        self.conv5 = nn.Conv2d(self.conv1_out_channels + self.conv2_out_channels + self.conv3_out_channels + self.conv4_out_channels, self.conv5_out_channels, kernel_size=3, stride=1, padding=1)  # 크기 유지

        # Pooling layer (최종 크기를 6x6으로 줄이기 위한 Adaptive Pooling)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.conv5_out_channels * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_cls),
        )

    def forward(self, x):
        # First layer
        x1 = self.conv1(x)

        # Second layer
        x2 = self.conv2(x1)

        # Concatenate outputs up to conv2
        x2_cat = torch.cat([x1, x2], dim=1)

        # Third layer
        x3 = F.relu(self.conv3(x2_cat))

        # Concatenate outputs up to conv3
        x3_cat = torch.cat([x2_cat, x3], dim=1)

        # Fourth layer
        x4 = F.relu(self.conv4(x3_cat))

        # Concatenate outputs up to conv4
        x4_cat = torch.cat([x3_cat, x4], dim=1)

        # Fifth layer
        x5 = F.relu(self.conv5(x4_cat))

        # Pooling
        x5 = self.pool(x5)

        # Flatten and classify
        x5 = torch.flatten(x5, 1)
        x = self.classifier(x5)
        return x


if __name__== '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DenseAlexNet(num_cls=10).to(device)
    print(model)

    # 데이터셋 전처리 및 데이터 증강
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    # CIFAR-10 데이터셋 로드
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    writer = SummaryWriter()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
    best_accuracy = 0.0
    early_stop_count = 0
    patience = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # 검증 루프
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            early_stop_count = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered.")
                break