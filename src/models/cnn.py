import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """Convolutional Neural Network for CIFAR-10"""
    
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # Conv layers with pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer
        self.fc = nn.Linear(128 * 4 * 4, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = self.fc(x)
        return x
