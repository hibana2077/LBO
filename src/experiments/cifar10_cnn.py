import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import load_cifar10
from models import ConvNet
from utils import get_optimizers, save_results

def experiment4_convnet_cifar10():
    """Experiment 4: Convolutional Neural Networks on CIFAR-10
    
    ConvNet with 3x convolution+pooling, 1x fully connected, with dropout
    
    Compares:
    - Adam vs. Adagrad vs. SGD (Nesterov momentum) vs. LBO
    
    Returns:
        list: Experiment results
    """
    print("Running Experiment 4: Convolutional Neural Network on CIFAR-10")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, _ = load_cifar10()
    
    # Initialize model
    model = ConvNet().to(device)
    
    # Get optimizers
    optimizers = get_optimizers(model.parameters(), 4)
    
    # Results storage
    results = []
    
    # Training
    num_epochs = 20
    
    for opt_name, optimizer in optimizers.items():
        print(f"Training with {opt_name}...")
        
        # Reset model
        model = ConvNet().to(device)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * data.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            
            results.append({
                'optimizer': opt_name,
                'epoch': epoch + 1,
                'loss': epoch_loss
            })
    
    save_results("convnet_cifar10", results)
    return results
