import torch
from tqdm import tqdm

from datasets import load_mnist
from models import VAE, vae_loss
from utils import get_optimizers, save_results

def experiment5_vae():
    """Experiment 5: Variational Auto-Encoder
    
    1 hidden layer (500 units, softplus), 50-dimensional Gaussian latent variable
    
    Compares:
    - Adam (with/without bias correction)
    - Adam with varying β₁, β₂, α
    - LBO
    
    Returns:
        list: Experiment results
    """
    print("Running Experiment 5: Variational Auto-Encoder")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data (using MNIST)
    train_loader, _ = load_mnist()
    
    # Initialize model
    model = VAE().to(device)
    
    # Get optimizers
    optimizers = get_optimizers(model.parameters(), 5)
    
    # Results storage
    results = []
    
    # Training
    num_epochs = 100  # We'll check at 10 and 100 epochs as specified
    check_points = [10, 100]
    
    for opt_name, optimizer in optimizers.items():
        print(f"Training with {opt_name}...")
        
        # Reset model
        model = VAE().to(device)
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                data = data.to(device)
                
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() / len(data)
            
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            
            if (epoch + 1) in check_points:
                results.append({
                    'optimizer': opt_name,
                    'epoch': epoch + 1,
                    'loss': epoch_loss
                })
    
    save_results("vae", results)
    return results
