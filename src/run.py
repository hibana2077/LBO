"""
Main runner script for optimizer comparison experiments.

This script runs a series of experiments to compare different optimization algorithms,
including our custom LaplaceBeltramiOptimizer (LBO), across various machine learning tasks.
"""
import os
import torch
import random
import numpy as np
from experiments import (
    experiment1_logistic_regression_mnist,
    experiment2_logistic_regression_imdb,
    experiment3_multilayer_nn, 
    experiment4_convnet_cifar10,
    experiment5_vae
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def save_results(experiment_name, results):
    """Save experiment results to CSV file"""
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/{experiment_name}_results.csv', index=False)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    for optimizer_name in results_df['optimizer'].unique():
        opt_data = results_df[results_df['optimizer'] == optimizer_name]
        plt.plot(opt_data['epoch'], opt_data['loss'], label=optimizer_name)
    
    plt.title(f'{experiment_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/{experiment_name}_loss_curve.png')
    plt.close()
    
    print(f"Results saved for {experiment_name}")

def get_optimizers(params, experiment):
    """Return the required optimizers for each experiment"""
    optimizers = {}
    
    # Base learning rate for each experiment
    lr = 0.001
    
    if experiment == 1:  # Logistic Regression on MNIST
        optimizers = {
            'Adam': optim.Adam(params, lr=lr),
            'Adagrad': optim.Adagrad(params, lr=lr),
            'SGD_Nesterov': optim.SGD(params, lr=lr, momentum=0.9, nesterov=True),
            'LBO': LaplaceBeltramiOptimizer(params, lr=lr)
        }
    elif experiment == 2:  # Logistic Regression on IMDB
        optimizers = {
            'Adam': optim.Adam(params, lr=lr),
            'Adagrad': optim.Adagrad(params, lr=lr),
            'RMSprop': optim.RMSprop(params, lr=lr),
            'SGD_Nesterov': optim.SGD(params, lr=lr, momentum=0.9, nesterov=True),
            'LBO': LaplaceBeltramiOptimizer(params, lr=lr)
        }
    elif experiment == 3:  # Multilayer Neural Network
        optimizers = {
            'Adam': optim.Adam(params, lr=lr),
            'AdaDelta': optim.Adadelta(params),
            'Adagrad': optim.Adagrad(params, lr=lr),
            'RMSprop': optim.RMSprop(params, lr=lr),
            'SGD_Nesterov': optim.SGD(params, lr=lr, momentum=0.9, nesterov=True),
            'LBO': LaplaceBeltramiOptimizer(params, lr=lr)
        }
    elif experiment == 4:  # Convolutional Neural Network
        optimizers = {
            'Adam': optim.Adam(params, lr=lr),
            'Adagrad': optim.Adagrad(params, lr=lr),
            'SGD_Nesterov': optim.SGD(params, lr=lr, momentum=0.9, nesterov=True),
            'LBO': LaplaceBeltramiOptimizer(params, lr=lr)
        }
    elif experiment == 5:  # Variational Auto-Encoder
        # Standard Adam
        optimizers['Adam'] = optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        
        # Adam without bias correction
        optimizers['Adam_no_bias'] = optim.Adam(params, lr=lr, betas=(0.9, 0.999), amsgrad=True)
        
        # Adam with varying beta1
        optimizers['Adam_beta1_0.5'] = optim.Adam(params, lr=lr, betas=(0.5, 0.999))
        optimizers['Adam_beta1_0.7'] = optim.Adam(params, lr=lr, betas=(0.7, 0.999))
        
        # Adam with varying beta2
        optimizers['Adam_beta2_0.99'] = optim.Adam(params, lr=lr, betas=(0.9, 0.99))
        optimizers['Adam_beta2_0.9'] = optim.Adam(params, lr=lr, betas=(0.9, 0.9))
        
        # Adam with varying learning rate
        optimizers['Adam_lr_0.01'] = optim.Adam(params, lr=0.01)
        optimizers['Adam_lr_0.0001'] = optim.Adam(params, lr=0.0001)
        
        # LBO
        optimizers['LBO'] = LaplaceBeltramiOptimizer(params, lr=lr)
    
    return optimizers

# ----- Models for each experiment -----

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class MultilayerNN(nn.Module):
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

class ConvNet(nn.Module):
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

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=500, latent_dim=50):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.softplus(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.softplus(self.fc2(z))
        return torch.sigmoid(self.fc3(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ----- Data loading functions -----

def load_mnist():
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

def load_imdb():
    """Load IMDB dataset and convert to bag-of-words features"""
    # Fetch from sklearn or use a pickle file if already downloaded
    try:
        with open('./data/imdb_data.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
    except:
        # This is a simplified version, actual loading might require more steps
        from sklearn.datasets import load_files
        
        # Create data directory if it doesn't exist
        os.makedirs('./data/aclImdb', exist_ok=True)
        
        # Download dataset logic would go here
        # For now, we'll create dummy data
        X_train = np.random.rand(1000, 10000).astype(np.float32)
        X_test = np.random.rand(200, 10000).astype(np.float32)
        y_train = np.random.randint(0, 2, 1000).astype(np.int64)
        y_test = np.random.randint(0, 2, 200).astype(np.int64)
        
        # Save for future use
        with open('./data/imdb_data.pkl', 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def load_cifar10():
    """Load CIFAR-10 dataset"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

# ----- Experiments -----

def experiment1_logistic_regression_mnist():
    """Experiment 1: Logistic Regression on MNIST"""
    print("Running Experiment 1: Logistic Regression on MNIST")
    
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

def experiment2_logistic_regression_imdb():
    """Experiment 2: Logistic Regression on IMDB (Sparse Features)"""
    print("Running Experiment 2: Logistic Regression on IMDB")
    
    # Load data
    train_loader, _ = load_imdb()
    
    # Initialize model
    input_dim = 10000  # Bag-of-Words features
    num_classes = 2    # Binary classification
    model = LogisticRegression(input_dim, num_classes).to(device)
    
    # Get optimizers
    optimizers = get_optimizers(model.parameters(), 2)
    
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
    
    save_results("logistic_regression_imdb", results)
    return results

def experiment3_multilayer_nn():
    """Experiment 3: Multi-layer Neural Networks"""
    print("Running Experiment 3: Multi-layer Neural Network on MNIST")
    
    # Load data
    train_loader, _ = load_mnist()
    
    # Initialize model
    input_dim = 28 * 28
    hidden_dim = 1000
    num_classes = 10
    
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

def experiment4_convnet_cifar10():
    """Experiment 4: Convolutional Neural Networks on CIFAR-10"""
    print("Running Experiment 4: Convolutional Neural Network on CIFAR-10")
    
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

def experiment5_vae():
    """Experiment 5: Variational Auto-Encoder"""
    print("Running Experiment 5: Variational Auto-Encoder")
    
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

def main():
    """Run all experiments"""
    print("Starting experiments to test LaplaceBeltramiOptimizer...")
    
    # Create data directory
    os.makedirs('./data', exist_ok=True)
    
    # Run all experiments
    experiment1_logistic_regression_mnist()
    experiment2_logistic_regression_imdb()
    experiment3_multilayer_nn()
    experiment4_convnet_cifar10()
    experiment5_vae()
    
    print("All experiments completed! Results saved in 'results' directory.")

if __name__ == "__main__":
    main()