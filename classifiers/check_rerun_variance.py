import os
import pickle # Added for loading .pkl files
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import auc

# Removed unused imports and find_best_model_path function

def load_roc_curves(filename):
    """
    Load ROC curves from a pickle file.

    Args:
        filename (str): Path to the saved file

    Returns:
        dict: Dictionary containing ROC curve data, or None if file not found.
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                rocs = pickle.load(f)
            print(f"ROC curves loaded from {filename}")
            return rocs
        except Exception as e:
            print(f"Error loading pickle file {filename}: {e}")
            return None
    else:
        print(f"Error: ROC curve file not found: {filename}")
        return None

def calculate_variance_from_loaded_rocs(rocs_data):
    """
    Calculates the standard deviation of AUC scores from pre-loaded ROC data.

    Args:
        rocs_data (dict): Dictionary loaded from the .pkl file.
                          Keys are parameter settings (e.g., '500', '1000'),
                          Values are lists of ROC data points from repetitions.
                          Each ROC data point should be a dict {'fpr': ..., 'tpr': ...}
                          or a tuple/list (fpr_array, tpr_array).

    Returns:
        tuple: (list of parameter labels, list of corresponding AUC std devs)
               Returns (None, None) if input is invalid.
    """
    if not rocs_data or not isinstance(rocs_data, dict):
        print("Error: Invalid or empty ROC data provided.")
        return None, None

    parameter_auc_std_devs = []
    parameter_labels = [] # Will store the keys from the rocs_data dict

    print("Calculating AUC variance from loaded ROC data...")
    # Iterate through each parameter setting (key in the loaded dictionary)
    # Sort keys numerically if they represent numbers, otherwise alphabetically
    try:
        # Attempt to sort keys as numbers
        sorted_keys = sorted(rocs_data.keys(), key=lambda x: int(re.sub(r'[^\d]', '', x))) # Extract numbers for sorting
    except ValueError:
        # Fallback to alphabetical sort if keys aren't purely numeric
        sorted_keys = sorted(rocs_data.keys())
        print("Warning: Non-numeric keys detected. Sorting alphabetically.")

    for param_key in tqdm(sorted_keys, desc="Parameter Settings"):
        roc_values = rocs_data[param_key]
        parameter_labels.append(str(param_key)) # Use the key as the label
        repetition_aucs = []

        if not isinstance(roc_values, list):
            print(f"Warning: Expected a list of ROC data for key '{param_key}', got {type(roc_values)}. Skipping.")
            parameter_auc_std_devs.append(np.nan)
            continue

        # Iterate through each repetition's ROC data for the current parameter
        for idx, roc_data in enumerate(roc_values):
            try:
                # Extract FPR and TPR
                if isinstance(roc_data, dict):
                    fpr = np.array(roc_data['fpr'])
                    tpr = np.array(roc_data['tpr'])
                elif isinstance(roc_data, (list, tuple)) and len(roc_data) >= 2:
                    fpr = np.array(roc_data[0])
                    tpr = np.array(roc_data[1])
                else:
                    print(f"Warning: Invalid roc_data format for {param_key}, repetition {idx+1}. Skipping.")
                    continue

                if fpr.size == 0 or tpr.size == 0 or fpr.size != tpr.size:
                     print(f"Warning: Empty or mismatched FPR/TPR for {param_key}, repetition {idx+1}. Skipping.")
                     continue

                # Ensure fpr is sorted for AUC calculation
                sort_indices = np.argsort(fpr)
                fpr_sorted = fpr[sort_indices]
                tpr_sorted = tpr[sort_indices]

                if len(np.unique(fpr_sorted)) >= 2: # Need at least 2 unique points for AUC
                    roc_auc = auc(fpr_sorted, tpr_sorted)
                    repetition_aucs.append(roc_auc)
                else:
                    print(f"Warning: Insufficient unique FPR points for AUC calculation in {param_key}, repetition {idx+1}. Skipping AUC.")

            except KeyError as e:
                 print(f"Warning: Missing key {e} in roc_data for {param_key}, repetition {idx+1}. Skipping.")
            except Exception as e:
                print(f"Error calculating AUC for {param_key}, repetition {idx+1}: {e}")

        # Calculate standard deviation for this parameter setting
        if len(repetition_aucs) >= 2:
            std_dev = np.std(repetition_aucs)
            parameter_auc_std_devs.append(std_dev)
        elif len(repetition_aucs) == 1:
            parameter_auc_std_devs.append(0.0)
            print(f"Warning: Only 1 valid AUC found for parameter {param_key}. Std Dev set to 0.")
        else:
            parameter_auc_std_devs.append(np.nan)
            print(f"Warning: No valid AUCs found for parameter {param_key}. Std Dev set to NaN.")

    if len(parameter_auc_std_devs) != len(rocs_data):
         print("Error: Mismatch between number of std devs calculated and number of parameters.")
         # This case might be complex if some keys were skipped entirely.
         # For simplicity, we'll return what we have, but a mismatch indicates issues.
         pass # Continue with potentially partial results

    return parameter_labels, parameter_auc_std_devs

def plot_variance(parameter_labels, std_devs, title='AUC Standard Deviation Across Repetitions'):
    """Plots the standard deviation against parameter settings."""
    if not parameter_labels or not std_devs or len(parameter_labels) != len(std_devs):
        print("Error: Invalid data provided for plotting.")
        return

    valid_indices = [i for i, sd in enumerate(std_devs) if not np.isnan(sd)]
    if not valid_indices:
        print("Error: No valid standard deviation values to plot.")
        return

    valid_labels = [parameter_labels[i] for i in valid_indices]
    valid_std_devs = [std_devs[i] for i in valid_indices]

    x_pos = np.arange(len(valid_labels)) # Use numerical positions for plotting

    plt.figure(figsize=(12, 6))
    plt.bar(x_pos, valid_std_devs, align='center', alpha=0.7)
    plt.xticks(x_pos, valid_labels, rotation=45, ha='right') # Set labels to parameter values
    plt.xlabel('Parameter Setting (n value)')
    plt.ylabel('Standard Deviation of Real-Life AUC')
    plt.title(title)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show() # Show the plot instead of saving

# --- Configuration ---
start_index = 2600
unique_n_vals = [1,2,3,5,10,20,30]
repetitions = 4 # Number of times the sweep was repeated
# --- End Configuration ---

if __name__ == "__main__":
    # Calculate the expected filename for the pre-calculated ROC curves
    # This logic should match the filename generation in eval_real_life_roc.py
    if not unique_n_vals:
        print("Error: unique_n_vals list is empty. Cannot determine filename.")
        exit()
    if repetitions <= 0:
         print("Error: repetitions must be greater than 0.")
         exit()

    end_index = start_index + len(unique_n_vals) * repetitions - 1
    roc_filename = f'classifiers/roc_curves_{start_index}-{end_index}.pkl'

    # Load the pre-calculated ROC data
    rocs_data = load_roc_curves(roc_filename)

    if rocs_data:
        # Calculate variance from the loaded data
        param_labels, auc_std_devs = calculate_variance_from_loaded_rocs(rocs_data)

        # Plot the results
        if param_labels is not None and auc_std_devs is not None:
            plot_variance(param_labels, auc_std_devs)
        else:
            print("Could not calculate variance from loaded data.")
    else:
        print(f"Failed to load ROC data from {roc_filename}. Cannot proceed.")