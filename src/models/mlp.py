import torch
import torch.nn as nn
import torch.nn.functional as F

class MultilayerNN(nn.Module):
    """2-hidden layer neural network with optional dropout"""
    
    def __init__(self, input_dim, hidden_dim=1000, num_classes=10, dropout_rate=0.5, use_dropout=True):
        super(MultilayerNN, self).__init__()
        self.use_dropout = use_dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc3(x)
        return x
