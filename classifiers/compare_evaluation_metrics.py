# this file re-plots training metrics from a (rurunohinohi) model and also finds the best model based on a target metric

import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configuration
metrics_dir = 'classifiers/evaluation_results'
metric_idxs = [1, 2, 3, 4, 10, 11, 12, 13, 14]
target_metric = 'rl-auc'  # Change this to any metric you want to optimize

# Function to load metrics using the unsafe YAML loader (since we trust the source)
def load_metrics(metrics_dir, model_idxs):
    all_metrics = {}
    
    for model_idx in model_idxs:
        file_path = os.path.join(metrics_dir, f'metrics_{model_idx}.yaml')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                # Using unsafe loader since you generated the YAML and trust it
                metrics = yaml.load(f, Loader=yaml.Loader)
                all_metrics[model_idx] = metrics
    
    return all_metrics

# Load all metrics
all_metrics = load_metrics(metrics_dir, metric_idxs)
print(f'available metrics: {all_metrics[metric_idxs[0]][0].keys()}')

# Plot metrics for a specific model
def plot_model_metrics(model_idx):
    if model_idx not in all_metrics:
        print(f"Model {model_idx} not found in metrics")
        return
    
    metrics = all_metrics[model_idx]
    epochs = range(1, len(metrics) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot loss metrics in the top subplot
    train_loss = [float(epoch_data.get('train_loss', 0)) for epoch_data in metrics]
    val_loss = [float(epoch_data.get('val_loss', 0)) for epoch_data in metrics]
    
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Metrics for Model {model_idx}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot all performance metrics in the bottom subplot
    ax2.plot(epochs, [float(epoch_data.get('rl-auc', 0)) for epoch_data in metrics], 'b-', label='RL-AUC')
    ax2.plot(epochs, [float(epoch_data.get('rl-f1', 0)) for epoch_data in metrics], 'g-', label='RL-F1')
    ax2.plot(epochs, [float(epoch_data.get('rl-precision', 0)) for epoch_data in metrics], 'r-', label='RL-Precision')
    ax2.plot(epochs, [float(epoch_data.get('rl-recall', 0)) for epoch_data in metrics], 'c-', label='RL-Recall')
    
    # Add validation metrics
    ax2.plot(epochs, [float(epoch_data.get('val_auc', 0)) for epoch_data in metrics], 'b--', label='Val-AUC')
    ax2.plot(epochs, [float(epoch_data.get('val_f1', 0)) for epoch_data in metrics], 'g--', label='Val-F1')
    ax2.plot(epochs, [float(epoch_data.get('val_precision', 0)) for epoch_data in metrics], 'r--', label='Val-Precision')
    ax2.plot(epochs, [float(epoch_data.get('val_recall', 0)) for epoch_data in metrics], 'c--', label='Val-Recall')
    
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric Value')
    ax2.set_title(f'Performance Metrics for Model {model_idx}')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Find best epochs and models for a specific metric
def find_best_models(all_metrics, metric_name):
    best_per_model = {}
    all_values = []
    
    for model_idx, metrics in all_metrics.items():
        best_epoch = 0
        best_value = -float('inf')
        
        for epoch, epoch_data in enumerate(metrics, 1):
            if metric_name in epoch_data:
                value = float(epoch_data[metric_name])
                if value > best_value:
                    best_value = value
                    best_epoch = epoch
        
        if best_epoch > 0:
            best_per_model[model_idx] = (best_epoch, best_value)
            all_values.append((model_idx, best_epoch, best_value))
    
    # Sort by metric value (worst to best)
    all_values.sort(key=lambda x: x[2])
    
    return best_per_model, all_values

# Plot model 14's metrics
# plot_model_metrics(14)

# Find and print the best models for the target metric
best_per_model, sorted_models = find_best_models(all_metrics, target_metric)

print(f"\nModels ranked by best {target_metric} (worst to best):")
for model_idx, epoch, value in sorted_models:
    print(f"Model {model_idx} (epoch {epoch}): {value:.4f}")

# Find the overall best model
if sorted_models:
    best_model, best_epoch, best_value = sorted_models[-1]
    print(f"\nBest overall model for {target_metric}: Model {best_model}, Epoch {best_epoch}, Value: {best_value:.4f}")
else:
    print(f"\nNo models found with metric: {target_metric}")