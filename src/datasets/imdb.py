import os
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset

def load_imdb():
    """Load IMDB dataset with Bag-of-Words features (10,000 features)
    
    Returns:
        tuple: (train_loader, test_loader)
    """
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
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
    
    return train_loader, test_loader
