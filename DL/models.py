import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_cls):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_cls),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetWithSkip1(nn.Module):
    def __init__(self, num_cls=1000):
        super(AlexNetWithSkip1, self).__init__()

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

    def forward(self, x):
        x = self.conv1(x)

        skip1_out = self.skip1(x)
        x = self.conv2(x)
        x = self.relu(self.conv3(x) + skip1_out)

        skip2_out = self.skip2(x)
        x = self.conv4(x)
        x = self.relu(self.conv5(x) + skip2_out)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class AlexNetWithSkip2(nn.Module):
    def __init__(self, num_cls=1000):
        super(AlexNetWithSkip2, self).__init__()

        # Feature extractor 정의
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.skip1 = self.skip_connection_pooling(3, 96, 13, 8)
        self.skip2 = self.skip_connection_pooling(96, 256, 3, 2)
        self.skip3 = self.skip_connection(256, 384)
        self.skip4 = self.skip_connection(384, 384)
        self.skip5 = self.skip_connection(384, 256)

        # Classifier 정의
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_cls),
        )

    def skip_connection_pooling(self, in_dim, out_dim, kernel_size, stride):
        sc = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_dim)
        )
        return sc

    def skip_connection(self, in_dim, out_dim):
        sc = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_dim)
        )
        return sc
    
    def forward(self, x):
        # conv1 + skip connection
        residual = self.skip1(x)
        x = self.conv1(x)
        x = F.relu(x + residual)

        # conv2 + skip connection
        residual = self.skip2(x)
        x = self.conv2(x)
        x = F.relu(x + residual)

        # conv3 + skip connection
        residual = self.skip3(x)
        x = self.conv3(x)
        x = F.relu(x + residual)

        # conv4 + skip connection
        residual = self.skip4(x)
        x = self.conv4(x)
        x = F.relu(x + residual)

        # conv5 + skip connection
        residual = self.skip5(x)
        x = self.conv5(x)
        x = F.relu(x + residual)

        # Pooling
        x = self.pool(x)

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x