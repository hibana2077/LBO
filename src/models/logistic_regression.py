import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    """L2-regularized multi-class logistic regression model"""
    
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)
