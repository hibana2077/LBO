import torch.optim as optim
from opt.lbo import LaplaceBeltramiOptimizer

def get_optimizers(params, experiment):
    """Return the required optimizers for each experiment
    
    Args:
        params: Model parameters to optimize
        experiment (int): The experiment number (1-5)
    
    Returns:
        dict: Dictionary of optimizer instances keyed by name
    """
    params = list(params)
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
