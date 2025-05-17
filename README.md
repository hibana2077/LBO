# Optimizer Testing Framework

This project provides a modular framework for testing different optimization algorithms across various machine learning tasks.

## Project Structure

The code is organized in a modular way to make it easy to modify and extend:

```sh
src/
├── datasets/                  # Dataset loading functions
│   ├── __init__.py
│   ├── mnist.py
│   ├── cifar10.py
│   └── imdb.py
├── models/                    # Neural network models
│   ├── __init__.py
│   ├── logistic_regression.py
│   ├── mlp.py
│   ├── cnn.py
│   └── vae.py
├── experiments/               # Experiment implementations
│   ├── __init__.py
│   ├── mnist_logistic.py
│   ├── imdb_logistic.py
│   ├── mnist_mlp.py
│   ├── cifar10_cnn.py
│   └── mnist_vae.py
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── optimizers.py
│   └── visualization.py
├── opt/                       # Custom optimizers
│   └── lbo.py
└── run_modular.py             # Main script to run experiments
```

## Running the Experiments

To run all experiments, use:

```bash
python src/run_modular.py
```

This will execute the following experiments:

1. **Logistic Regression on MNIST**
   - Model: L2-regularized multi-class logistic regression
   - Optimizers: Adam, Adagrad, SGD with Nesterov momentum, LBO

2. **Logistic Regression on IMDB (Sparse Features)**
   - Dataset: IMDB with Bag-of-Words (10,000 features)
   - Optimizers: Adam, Adagrad, RMSProp, SGD with Nesterov momentum, LBO

3. **Multi-layer Neural Networks on MNIST**
   - Model: 2 hidden layers (1000 units each), with and without dropout
   - Optimizers: Adam, AdaDelta, Adagrad, RMSProp, SGD with Nesterov momentum, LBO

4. **Convolutional Neural Networks on CIFAR-10**
   - Model: 3 convolutional layers with pooling, 1 fully connected layer
   - Optimizers: Adam, Adagrad, SGD with Nesterov momentum, LBO

5. **Variational Auto-Encoder on MNIST**
   - Model: 1 hidden layer (500 units), 50-dimensional latent space
   - Optimizers: Adam variants (with/without bias correction, varying β₁, β₂, α), LBO

## Modifying Experiments

To modify an experiment:

1. Edit the corresponding file in the `experiments/` directory
2. To change model architecture, modify the model classes in the `models/` directory
3. To change optimizer configurations, update the `get_optimizers()` function in `utils/optimizers.py`

## Results

Experiment results are saved in the `results/` directory as:

- CSV files with loss values for each optimizer
- PNG plots comparing optimizer performance

## Requirements

See `requirements.txt` for the necessary dependencies.
