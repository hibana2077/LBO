import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import load_mnist
from models import LogisticRegression
from utils import get_optimizers, save_results

def experiment1_logistic_regression_mnist():
    """Experiment 1: Logistic Regression on MNIST
    
    Compares:
    - Adam vs. Adagrad vs. SGD (Nesterov momentum) vs. LBO
    
    Returns:
        list: Experiment results
    """
    print("Running Experiment 1: Logistic Regression on MNIST")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, _ = load_mnist()
    
    # Initialize model
    input_dim = 28 * 28
    num_classes = 10
    model = LogisticRegression(input_dim, num_classes).to(device)
    
    # Get optimizers
    optimizers = get_optimizers(model.parameters(), 1)
    
    # Results storage
    results = []
    
    # Training
    num_epochs = 20
    
    for opt_name, optimizer in optimizers.items():
        print(f"Training with {opt_name}...")
        
        # Reset model
        model = LogisticRegression(input_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                data, target = data.to(device), target.to(device)
                data = data.view(-1, input_dim)
                
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
    
    save_results("logistic_regression_mnist", results)
    return results
