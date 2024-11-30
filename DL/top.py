import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from AlexNet_base import AlexNet

def save_confusion_matrix(model, dataloader, device, class_names, save_path="confusion_matrix.png"):
    """
    Confusion Matrix를 계산하고 저장합니다.
    Args:
        model: PyTorch 모델.
        dataloader: 테스트 데이터 로더.
        device: 'cuda' 또는 'cpu'.
        class_names: 클래스 이름 리스트.
        save_path: Confusion Matrix를 저장할 경로.
    """
    model.eval()  # 평가 모드
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 그래디언트 계산 비활성화
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 모델의 예측값
            _, preds = torch.max(outputs, 1)  # Top-1 예측값
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix 생성
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    
    # Confusion Matrix 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)  # 이미지 저장
    plt.close()
    print(f"Confusion Matrix saved to {save_path}")

# 예시 사용법
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델과 테스트 데이터 로더를 정의하세요
model = AlexNet(num_cls=10).to(device)  
model.load_state_dict(torch.load("best_model_base.pth"))

transform_test = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# CIFAR-10 클래스 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Confusion Matrix 저장
save_confusion_matrix(model, test_loader, device, class_names, save_path="confusion_matrix.png")
