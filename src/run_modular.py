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
def set_seed(seed=42):
    """Set random seeds for reproducibility across libraries"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """Run all experiments"""
    print("Starting experiments to test LaplaceBeltramiOptimizer...")
    
    # Set random seeds
    set_seed(42)
    
    # Create data directory
    os.makedirs('./data', exist_ok=True)
    
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    # Print device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run all experiments
    experiment1_logistic_regression_mnist()
    experiment2_logistic_regression_imdb()
    experiment3_multilayer_nn()
    experiment4_convnet_cifar10()
    experiment5_vae()
    
    print("All experiments completed! Results saved in 'results' directory.")

if __name__ == "__main__":
    main()
