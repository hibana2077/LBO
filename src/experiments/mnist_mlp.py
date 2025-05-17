import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import load_mnist
from models import MultilayerNN
from utils import get_optimizers, save_results

def experiment3_multilayer_nn():
    """Experiment 3: Multi-layer Neural Networks on MNIST
    
    Fully connected NN with 2 hidden layers (1000 units each, ReLU)
    Tests with and without dropout
    
    Compares:
    - Adam vs. AdaDelta vs. Adagrad vs. RMSProp vs. SGD (Nesterov momentum) vs. LBO
    
    Returns:
        dict: Results for with_dropout and without_dropout configurations
    """
    print("Running Experiment 3: Multi-layer Neural Network on MNIST")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, _ = load_mnist()
    
    # Initialize model
    input_dim = 28 * 28
    hidden_dim = 1000
    num_classes = 10
    
    # Store results for both dropout configurations
    all_results = {}
    
    # Test with and without dropout
    for use_dropout in [True, False]:
        dropout_str = "with_dropout" if use_dropout else "without_dropout"
        print(f"Training {dropout_str}...")
        
        model = MultilayerNN(input_dim, hidden_dim, num_classes, use_dropout=use_dropout).to(device)
        
        # Get optimizers
        optimizers = get_optimizers(model.parameters(), 3)
        
        # Results storage
        results = []
        
        # Training
        num_epochs = 20
        
        for opt_name, optimizer in optimizers.items():
            print(f"Training with {opt_name}...")
            
            # Reset model
            model = MultilayerNN(input_dim, hidden_dim, num_classes, use_dropout=use_dropout).to(device)
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
                    'loss': epoch_loss,
                    'dropout': use_dropout
                })
        
        save_results(f"multilayer_nn_{dropout_str}", results)
        all_results[dropout_str] = results
        
    return all_results
