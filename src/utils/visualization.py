import os
import pandas as pd
import matplotlib.pyplot as plt

def save_results(experiment_name, results):
    """Save experiment results to CSV file and plot loss curves
    
    Args:
        experiment_name (str): Name of the experiment
        results (list): List of dictionaries with experiment results
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save to CSV
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
