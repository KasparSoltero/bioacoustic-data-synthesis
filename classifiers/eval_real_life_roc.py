# this file gets roc curve for a given model (using real life little owl dataset n~70)

import pickle
import os
import torch
import yaml
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from tqdm import tqdm # Import tqdm

from kaytoo_small import BirdSoundModel, TrainingParameters, DefaultAudio, FilePaths, real_life_evaluate_with_roc, test_inference

def calculate_roc_curves(experiment_dirs, experiment_caption_prefix, unique_n_vals, repetitions, use_last=False):
    """
    Calculates ROC curves, grouping results by unique_n_vals across repetitions.

    Args:
        experiment_dirs (list): List of all experiment directory names (e.g., ['Exp_2300', 'Exp_2301', ...]).
                                It's assumed these are generated in a consistent, sortable manner.
        experiment_caption_prefix (str): Prefix for the legend labels.
        unique_n_vals (list): List of unique 'n' values (e.g., [1000, 5000, 10000]).
        repetitions (int): Number of experiment directories corresponding to each unique_n_val.
        use_last (bool): If True, only use the 'last.ckpt' checkpoint from each experiment. Defaults to False.
    """

    # Determine filename based on experiment range
    if not experiment_dirs:
        print("Warning: experiment_dirs list is empty. Cannot determine filename.")
        filename = 'roc_curves_unknown_range.pkl'
    else:
        # Extract numbers and find min/max
        exp_indices = sorted([int(d.split('_')[1]) for d in experiment_dirs])
        start_index = exp_indices[0]
        end_index = exp_indices[-1]
        base_filename = f'classifiers/roc_curves_{start_index}-{end_index}'
        if use_last:
            filename = f'{base_filename}_last.pkl'
        else:
            filename = f'{base_filename}.pkl'

    if os.path.exists(filename):
        print(f"Warning: {filename} already exists. Please delete it or choose a different name.")
        return

    # Load audio configuration
    audio_cfg = DefaultAudio()
    # Load use case
    with open('classifiers/use_case.yaml') as f:
        use_case = yaml.safe_load(f)
    train_cfg = TrainingParameters(options=use_case)
    paths = FilePaths(use_case)

    # single consensus roc plot
    # model_path = 'classifiers/evaluation_results/Exp_12/Results/binary_classifier-epoch=14-val_auc=0.858.ckpt'
    # bird_model = BirdSoundModel(train_cfg, audio_cfg, paths, in_channels=3)
    # model_state_dict = torch.load(model_path)
    # bird_model.load_state_dict(model_state_dict)

    # # here we load the model and evaluate it on the real life dataset
    # bird_model.eval()
    # model_path2 = 'classifiers/evaluation_results/Exp_14/Results/binary_classifier-epoch=15-val_auc=0.873.ckpt'
    # bird_model2 = BirdSoundModel(train_cfg, audio_cfg, paths, in_channels=3)
    # model_state_dict2 = torch.load(model_path2)
    # bird_model2.load_state_dict(model_state_dict2)
    # bird_model2.eval()
    # bird_models = [bird_model, bird_model2]
    # print('Model loaded')
    # eval_dir = paths.EVAL_DIR
    # metrics = real_life_evaluate_with_roc(bird_models, eval_dir, audio_cfg, consensus=True)

    rocs = {} # Dictionary to store ROC data, keyed by n_val

    if len(experiment_dirs) != len(unique_n_vals) * repetitions:
        raise ValueError("The total number of experiment_dirs must equal len(unique_n_vals) * repetitions.")

    # Initialize keys in the rocs dictionary first
    for n_val in unique_n_vals:
        roc_key = experiment_caption_prefix + str(n_val)
        rocs[roc_key] = []

    # Iterate through experiment directories with a progress bar
    num_dirs = len(experiment_dirs)
    for index, experiment_dir in tqdm(enumerate(experiment_dirs), total=num_dirs, desc="Calculating ROC curves"):
        # Determine the corresponding n_val and roc_key based on the index
        # This assumes the experiment_dirs list is ordered correctly (grouped by n_val, then repetition)
        # Example: If unique_n_vals=[1k, 5k, 10k] and reps=4,
        # indices 0-3 -> 1k, 4-7 -> 5k, 8-11 -> 10k
        n_val_index = index // repetitions # Integer division gives the index within unique_n_vals
        if n_val_index >= len(unique_n_vals):
             print(f"Warning: Index {index} out of bounds for unique_n_vals. Skipping.")
             continue
        n_val = unique_n_vals[n_val_index]
        roc_key = experiment_caption_prefix + str(n_val)

        results_dir = Path('classifiers/evaluation_results') / experiment_dir / 'Results'

        if not results_dir.exists():
            # tqdm handles printing alongside the bar, but keep warnings concise
            tqdm.write(f"Warning: Results dir not found, skipping: {results_dir}")
            continue

        model_paths = []
        try:
            if use_last:
                last_ckpt_path = results_dir / 'last.ckpt'
                if last_ckpt_path.exists():
                    model_paths = ['last.ckpt']
                else:
                    tqdm.write(f"Warning: 'last.ckpt' not found in {results_dir}, skipping experiment.")
                    continue # Skip this experiment dir if last.ckpt is expected but not found
            else:
                # Original logic: find all checkpoints matching the pattern
                model_paths = [f for f in os.listdir(results_dir) if f.startswith('binary_classifier-epoch=') and f.endswith('.ckpt')]
        except FileNotFoundError:
             tqdm.write(f"Warning: Error accessing or listing files in {results_dir}, skipping.")
             continue
        except Exception as e:
             tqdm.write(f"Warning: An unexpected error occurred listing files in {results_dir}: {e}. Skipping.")
             continue

        if not model_paths:
            # tqdm.write(f"Warning: No model checkpoints found in {results_dir}") # Optional: can be verbose
            pass # Silently continue if no models found

        # Process each model found in the directory
        for model_path in model_paths:
            full_model_path = results_dir / model_path
            try:
                bird_model = BirdSoundModel(train_cfg, audio_cfg, paths, in_channels=3)
                # Load model state dict - ensure loading to the correct device if necessary (e.g., CPU)
                # model_state_dict = torch.load(full_model_path, map_location=torch.device('cpu'))
                model_state_dict = torch.load(full_model_path) # Assuming model runs on available device
                bird_model.load_state_dict(model_state_dict)
                bird_model.eval()
                eval_dir = paths.EVAL_DIR
                # Assuming real_life_evaluate_with_roc returns a dict with 'roc_data'
                metrics = real_life_evaluate_with_roc(bird_model, eval_dir, audio_cfg, plot=False)
                if 'roc_data' in metrics:
                    # Check if roc_key exists, though it should have been initialized
                    if roc_key in rocs:
                        rocs[roc_key].append(metrics['roc_data']) # Append ROC data (fpr, tpr, thresholds)
                    else:
                        tqdm.write(f"Warning: roc_key '{roc_key}' not initialized. This shouldn't happen.")
                else:
                     tqdm.write(f"Warning: 'roc_data' not found for model {model_path} in {experiment_dir}")

            except Exception as e:
                print(e)
                tqdm.write(f"Error processing model {full_model_path}: {e}")

    # Save the aggregated ROC curves to the dynamically named file
    save_roc_curves(rocs, filename=filename)

def save_roc_curves(rocs, filename='roc_curves.pkl'):
    """
    Save ROC curves to a pickle file.
    
    Args:
        rocs (dict): Dictionary containing ROC curve data
        filename (str): Path to save the file
    """
    # check if file already exists
    if os.path.exists(filename):
        print(f'Error : {filename} already exists. Please delete it or choose a different name.')
        return
    with open(filename, 'wb') as f:
        pickle.dump(rocs, f)
    print(f"ROC curves saved to {filename}")

def load_roc_curves(filename):
    """
    Load ROC curves from a pickle file.
    
    Args:
        filename (str): Path to the saved file
        
    Returns:
        dict: Dictionary containing ROC curve data
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            rocs = pickle.load(f)
        print(f"ROC curves loaded from {filename}")
        return rocs
    else:
        print(f"File {filename} not found.")
        return None

def plot_all_roc_curves(rocs):
    plt.figure(figsize=(12, 10))
    for i, (experiment, roc_values) in enumerate(rocs.items()):
        color = colors[i % len(colors)]
        
        # Plot each ROC curve for this experiment
        for j, roc_data in enumerate(roc_values):
            # Handle different possible formats of roc_data
            if isinstance(roc_data, dict):
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
            else:  # Assume it's a tuple/list with fpr, tpr, thresholds
                fpr, tpr = roc_data[0], roc_data[1]
            
            # Use consistent color with some transparency for each experiment
            alpha = 0.7
            plt.plot(fpr, tpr, color=color, alpha=alpha, linewidth=1)
            
            # Only add to legend for the first curve in each experiment
            if j == 0:
                plt.plot([], [], color=color, label=experiment)
    
    # Add reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Training Set Sizes')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_roc_curves_with_bounds(rocs, colors, repetitions, top_n_per_rep=None, legend_title='Training Set Size'):
    """
    Plots ROC curves for different experiments (e.g., training set sizes 'n').
    For each experiment (group of repetitions), it selects the top N checkpoints
    from each repetition based on AUC, then calculates and plots the mean ROC curve
    of these selected checkpoints, along with a shaded area representing their
    minimum and maximum TPR values at each FPR point.

    Args:
        rocs (dict): Dictionary where keys are experiment labels (e.g., 'n=1000')
                     and values are lists of ROC data points. The list for each
                     experiment label contains ROC data from all repetitions,
                     ordered sequentially (rep1_ckpt1, rep1_ckpt2, ..., rep2_ckpt1, ...).
        colors (list): A list of colors to cycle through for different experiments.
        repetitions (int): The number of repetitions grouped under each experiment label.
        top_n_per_rep (int, optional): If provided, selects the top 'top_n_per_rep'
                                       checkpoints (by AUC) from *each* repetition
                                       to include in the bounds/mean calculation.
                                       Defaults to None (use all checkpoints).
        legend_title (str): Title for the plot legend.
    """
    CKPTS_PER_REP = 4 # Constant: Number of checkpoints saved per repetition/experiment run

    if top_n_per_rep is not None and isinstance(top_n_per_rep, int) and top_n_per_rep > 0:
        if top_n_per_rep > CKPTS_PER_REP:
            print(f"Warning: top_n_per_rep ({top_n_per_rep}) is greater than CKPTS_PER_REP ({CKPTS_PER_REP}). Using all {CKPTS_PER_REP} checkpoints per repetition.")
            top_n_per_rep = CKPTS_PER_REP
        print(f"Selecting top {top_n_per_rep} checkpoints per repetition (by AUC) for bounds/mean calculation.")
    elif top_n_per_rep is not None:
        print(f"Warning: Invalid value for top_n_per_rep ({top_n_per_rep}). Using all checkpoints.")
        top_n_per_rep = None # Reset to default (use all)

    plt.figure(figsize=(9, 8))
    # Common x-axis (FPR) points for interpolation
    common_fpr = np.linspace(0, 1, 1000)

    for i, (experiment, roc_values) in enumerate(rocs.items()):
        color = colors[i % len(colors)]

        if not roc_values:
            print(f"Warning: No ROC data found for experiment '{experiment}'. Skipping.")
            continue

        # Store data for selected checkpoints across all repetitions
        selected_interpolated_data = [] # List of tuples: (auc_value, interpolated_tpr_array)
        all_individual_aucs = [] # Store AUCs of *all* processed checkpoints for mean calculation comparison

        expected_total_curves = repetitions * CKPTS_PER_REP
        if len(roc_values) != expected_total_curves:
            print(f"Warning: Experiment '{experiment}' has {len(roc_values)} ROC curves, but expected {expected_total_curves} ({repetitions} reps * {CKPTS_PER_REP} ckpts/rep). Proceeding, but results might be skewed.")
            # Adjust repetitions if possible, or handle potential errors later
            actual_repetitions = len(roc_values) // CKPTS_PER_REP
            if len(roc_values) % CKPTS_PER_REP != 0:
                 print(f"Error: Number of curves ({len(roc_values)}) not divisible by checkpoints per rep ({CKPTS_PER_REP}). Cannot reliably process experiment '{experiment}'. Skipping.")
                 continue
            print(f"Adjusting effective repetitions to {actual_repetitions} for '{experiment}'.")
        else:
            actual_repetitions = repetitions


        # Iterate through each repetition's data
        for rep_idx in range(actual_repetitions):
            start_idx = rep_idx * CKPTS_PER_REP
            end_idx = start_idx + CKPTS_PER_REP
            roc_chunk = roc_values[start_idx:end_idx]

            chunk_interpolated_data = [] # Data for this repetition's checkpoints

            # Process each checkpoint within the repetition
            for ckpt_idx, roc_data in enumerate(roc_chunk):
                global_idx = start_idx + ckpt_idx # For consistent logging
                # --- Extract FPR and TPR ---
                try:
                    if isinstance(roc_data, dict):
                        fpr = np.array(roc_data['fpr'])
                        tpr = np.array(roc_data['tpr'])
                    elif isinstance(roc_data, (list, tuple)) and len(roc_data) >= 2:
                        fpr = np.array(roc_data[0])
                        tpr = np.array(roc_data[1])
                    else:
                        print(f"Warning: Invalid roc_data format for {experiment}, rep {rep_idx}, ckpt {ckpt_idx} (global idx {global_idx}). Skipping.")
                        continue

                    if fpr.size == 0 or tpr.size == 0 or fpr.size != tpr.size:
                         print(f"Warning: Empty or mismatched FPR/TPR for {experiment}, rep {rep_idx}, ckpt {ckpt_idx} (global idx {global_idx}). Skipping.")
                         continue

                except Exception as e:
                    print(f"Error processing ROC data for {experiment}, rep {rep_idx}, ckpt {ckpt_idx} (global idx {global_idx}): {e}. Skipping.")
                    continue

                # --- Calculate AUC for sorting ---
                sort_indices = np.argsort(fpr)
                fpr_sorted = fpr[sort_indices]
                tpr_sorted = tpr[sort_indices]

                if len(np.unique(fpr_sorted)) < 2:
                    print(f"Warning: Insufficient unique FPR values for AUC calculation in {experiment}, rep {rep_idx}, ckpt {ckpt_idx} (global idx {global_idx}). Assigning AUC=0.")
                    individual_auc = 0.0
                else:
                    individual_auc = auc(fpr_sorted, tpr_sorted)
                all_individual_aucs.append(individual_auc) # Store all AUCs

                # --- Interpolate TPR onto common FPR axis ---
                fpr_interp = np.concatenate(([0], fpr_sorted, [1]))
                tpr_interp = np.concatenate(([0], tpr_sorted, [1]))
                unique_indices = np.unique(fpr_interp, return_index=True)[1]
                fpr_interp = fpr_interp[unique_indices]
                tpr_interp = tpr_interp[unique_indices]

                interpolated_tpr = np.interp(common_fpr, fpr_interp, tpr_interp)
                chunk_interpolated_data.append((individual_auc, interpolated_tpr))

            # --- Select top N from this repetition's chunk ---
            if not chunk_interpolated_data:
                print(f"Warning: No valid curves processed for {experiment}, repetition {rep_idx}. Skipping this repetition.")
                continue

            num_in_chunk = len(chunk_interpolated_data)
            if top_n_per_rep is not None and num_in_chunk > top_n_per_rep:
                # Sort this chunk by AUC descending
                chunk_interpolated_data.sort(key=lambda x: x[0], reverse=True)
                selected_chunk_data = chunk_interpolated_data[:top_n_per_rep]
                print(f"  Selected top {len(selected_chunk_data)} from rep {rep_idx} for {experiment}. AUCs: {[f'{d[0]:.3f}' for d in selected_chunk_data]}")
            else:
                selected_chunk_data = chunk_interpolated_data # Use all valid curves from this chunk

            selected_interpolated_data.extend(selected_chunk_data) # Add selected curves to the main list for the experiment

        # --- Check if any curves were selected across all repetitions ---
        if not selected_interpolated_data:
             print(f"Error: No valid ROC curves selected for experiment '{experiment}' after processing all repetitions. Cannot plot.")
             continue

        # Extract the TPR arrays from the final selected data for this experiment
        interpolated_tprs_selected = np.array([data[1] for data in selected_interpolated_data])

        # --- Calculate Min, Max, Mean TPR based on the selected curves ---
        tpr_min = np.min(interpolated_tprs_selected, axis=0)
        tpr_max = np.max(interpolated_tprs_selected, axis=0)
        tpr_mean = np.mean(interpolated_tprs_selected, axis=0)

        # --- Calculate AUC of the mean curve ---
        mean_auc = auc(common_fpr, tpr_mean) # AUC of the averaged curve

        # --- Calculate and print comparison ---
        if all_individual_aucs: # Check if we have any individual AUCs from processed checkpoints
            mean_of_all_individual_aucs = np.mean(all_individual_aucs)
            selected_aucs = [data[0] for data in selected_interpolated_data]
            mean_of_selected_aucs = np.mean(selected_aucs) if selected_aucs else 0.0

            print(f"--- Experiment {experiment} AUC Check ---")
            print(f"  AUC of the mean ROC curve (calculated from selected curves): {mean_auc:.4f}")
            print(f"  Mean of AUCs for *selected* curves:                      {mean_of_selected_aucs:.4f}")
            print(f"  Mean of AUCs for *all* processed curves:                 {mean_of_all_individual_aucs:.4f}")
            # diff = abs(mean_auc - mean_of_individual_aucs) # Original comparison might be less relevant now
            # print(f"  Difference (Mean ROC AUC vs Mean Individual AUC): {diff:.4f}")
            print(f"------------------------------------")
        else:
            print(f"--- Experiment {experiment}: No individual AUCs calculated or selected. ---")
        # --- END Comparison ---

        num_curves_in_calc = len(selected_interpolated_data)
        print(f'Plotting mean ROC for {experiment} (calculated from {num_curves_in_calc} curves, AUC displayed: {mean_auc:.3f}) with bounds.')

        # --- Plotting ---
        # Plot the mean ROC curve
        plt.plot(common_fpr, tpr_mean, color=color,
                 label=f'{experiment} (AUC: {mean_auc:.3f})',
                 linewidth=2)

        # Plot the shaded area for bounds
        plt.fill_between(common_fpr, tpr_min, tpr_max, color=color, alpha=0.2, label=f'_Bounds {experiment}') # Underscore hides from legend

    # --- Final Plot Adjustments ---
    # Add reference line (random classifier)
    plt.plot([0, 1], [0, 1], linestyle='--', color='#666', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05]) # Slight margin at the top
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    # Adjust legend to avoid duplicate entries from fill_between
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    # Filter out the hidden '_Bounds' labels
    filtered_by_label = {label: handle for label, handle in by_label.items() if not label.startswith('_')}
    plt.legend(filtered_by_label.values(), filtered_by_label.keys(), loc="lower right", fontsize=10, title=legend_title)
    plt.grid(True)
    plt.tight_layout()

def plot_rocs_sidebyside(rocs1, rocs2, colors1, colors2, legend_title1, legend_title2, limit_curves=None):
    """
    Plots two sets of ROC curves with bounds side-by-side on a shared y-axis.

    Args:
        rocs1 (dict): Dictionary for the first set of ROC data. Keys are experiment names,
                      values are lists of ROC curve data (e.g., [(fpr, tpr), ...]).
        rocs2 (dict): Dictionary for the second set of ROC data.
        colors1 (list): List of colors for the first plot.
        colors2 (list): List of colors for the second plot.
        legend_title1 (str): Title for the legend of the first plot.
        legend_title2 (str): Title for the legend of the second plot.
        limit_curves (int, optional): Maximum number of curves (sorted by AUC) to use
                                     for calculating bounds per experiment. Defaults to None (use all).
    """

    if limit_curves is not None and isinstance(limit_curves, int):
        print(f"Limiting to top {limit_curves} curves per experiment for bounds calculation.")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True) # Create 1x2 subplots, share y-axis
    fontsize = 18
    common_fpr = np.linspace(0, 1, 1000) # Common x-axis points for interpolation

    # --- Helper function to plot one set of ROCs ---
    def _plot_single_roc_set(ax, rocs, colors, legend_title):
        for i, (experiment, roc_values) in enumerate(rocs.items()):
            color = colors[i % len(colors)]

            # Arrays to store interpolated TPR values and AUCs
            interpolated_data = [] # Stores (auc, interpolated_tpr) tuples

            if not roc_values:
                print(f"Warning: No ROC curves found for experiment '{experiment}' in this set.")
                continue

            # Process each model's ROC curve within the experiment
            for j, roc_data in enumerate(roc_values):
                # Extract FPR and TPR
                if isinstance(roc_data, dict):
                    fpr = roc_data.get('fpr', [])
                    tpr = roc_data.get('tpr', [])
                elif isinstance(roc_data, (list, tuple)) and len(roc_data) >= 2:
                    fpr, tpr = roc_data[0], roc_data[1]
                else:
                    print(f"Warning: Skipping invalid roc_data format in experiment '{experiment}', item {j}")
                    continue

                fpr_orig = np.array(fpr) # Keep original arrays
                tpr_orig = np.array(tpr)
                if fpr_orig.size == 0 or tpr_orig.size == 0 or fpr_orig.size != tpr_orig.size:
                    print(f"Warning: Skipping empty or mismatched FPR/TPR in experiment '{experiment}', item {j}")
                    continue

                # --- AUC Calculation (using sorted data, as before) ---
                sort_indices = np.argsort(fpr_orig)
                fpr_sorted = fpr_orig[sort_indices]
                tpr_sorted = tpr_orig[sort_indices]

                if len(np.unique(fpr_sorted)) < 2:
                     print(f"Warning: Skipping curve with insufficient unique FPR values for AUC calculation in experiment '{experiment}', item {j}")
                     individual_auc = 0.0
                else:
                    individual_auc = auc(fpr_sorted, tpr_sorted) # AUC uses sorted values

                # --- Interpolation for Plotting (using ORIGINAL potentially unsorted data) ---
                # Use the original fpr_orig, tpr_orig arrays here to match the user's single plot function.
                # Note: This deviates from np.interp's requirement that the x-array (fpr_orig) be monotonic.
                interpolated_tpr = np.interp(common_fpr, fpr_orig, tpr_orig)
                # --- End Modification ---

                interpolated_tpr[0] = 0.0  # Force start at 0
                interpolated_data.append((individual_auc, interpolated_tpr)) # Store AUC (from sorted) and interpolated TPR (from original)

            # --- Start: Limit functionality ---
            num_curves_processed = len(interpolated_data)
            if limit_curves is not None and isinstance(limit_curves, int) and limit_curves > 0 and num_curves_processed > limit_curves:
                interpolated_data.sort(key=lambda x: x[0], reverse=True) # Sort by AUC descending
                selected_data = interpolated_data[:limit_curves]
            else:
                selected_data = interpolated_data

            # Handle case where no curves are left after filtering or initially
            if not selected_data:
                 print(f"Warning: No valid ROC curves left to plot for experiment '{experiment}' after processing/filtering.")
                 continue # Skip to the next experiment

            interpolated_tprs_for_avg = np.array([data[1] for data in selected_data])
             # --- End: Limit functionality ---


            if interpolated_tprs_for_avg.size == 0:
                 print(f"Warning: No interpolated TPRs to calculate bounds/mean for experiment '{experiment}'.")
                 continue

            # Find min and max TPR values at each FPR point
            tpr_min = np.min(interpolated_tprs_for_avg, axis=0)
            tpr_max = np.max(interpolated_tprs_for_avg, axis=0)

            # Calculate mean TPR (for the central line)
            tpr_mean = np.mean(interpolated_tprs_for_avg, axis=0)

            mean_auc = auc(common_fpr, tpr_mean)

            # Plot the mean line
            print(f'Plotting mean ROC curve for {experiment} with AUC: {mean_auc:.2f}')
            ax.plot(common_fpr, tpr_mean, color=color,
                    label=f'{experiment} ({mean_auc:.2f})',
                    linewidth=2)

            # Plot the filled area between min and max
            ax.fill_between(common_fpr, tpr_min, tpr_max, color=color, alpha=0.2)

        # Add reference line (random classifier)
        ax.plot([0, 1], [0, 1], linestyle='--', color='#666')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0]) # Ensure y-lim is consistent
        
        ax.set_xlabel('False Positive Rate', fontsize=fontsize)
        ax.legend(loc="lower right", fontsize=fontsize, title=legend_title, title_fontsize=fontsize-2)
        # tick fontsize
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.grid(True)

    # --- Plotting ---
    # Plot left side
    _plot_single_roc_set(axes[0], rocs1, colors1, legend_title1)
    axes[0].set_ylabel('True Positive Rate', fontsize=fontsize) # Set Y label only on the left plot

    # Plot right side
    _plot_single_roc_set(axes[1], rocs2, colors2, legend_title2)
    # axes[1].tick_params(axis='y', labelleft=False) # Hide y-axis tick labels on the right plot (handled by sharey=True)

    # --- Final Touches ---
    # Add titles if desired (optional)
    # axes[0].set_title('ROC Curves - Set 1', fontsize=14)
    # axes[1].set_title('ROC Curves - Set 2', fontsize=14)

    plt.tight_layout() # Adjust layout to prevent overlap

def relabel_rocs(rocs, experiment_caption_prefix, experiment_n_vals):
    new_rocs = {}
    saved_keys = list(rocs.keys())
    for i, key in enumerate(saved_keys):
        n_val = experiment_n_vals[i]
        new_rocs[experiment_caption_prefix + str(n_val)] = rocs[key]
    return new_rocs

def generate_experiment_dir_strings(start_index, unique_n_vals, repetitions):
    """
    Generates the list of experiment directory names, grouped by unique_n_vals first, then by repetition.

    Args:
        start_index (int): The starting index for the experiment names (e.g., 2300).
        unique_n_vals (list): List of unique 'n' values.
        repetitions (int): Number of repetitions for each 'n' value.

    Returns:
        list: List of experiment directory names (e.g., ['Exp_2300', 'Exp_2303', 'Exp_2306', 'Exp_2309', 'Exp_2301', ...]).
    """
    experiment_dirs = []
    num_unique_vals = len(unique_n_vals)
    # The logic iterates through unique values first, then repetitions.
    for val_idx in range(num_unique_vals):
        for rep_idx in range(repetitions):
            # Calculate the experiment index based on start, repetition, and value position
            # This ensures experiments for the same n_val but different repetitions are grouped.
            exp_index = start_index + val_idx + (rep_idx * num_unique_vals)
            experiment_dirs.append(f'Exp_{exp_index}')
    return experiment_dirs


# Example usage for averaging repetitions:
# Define the parameters for generated experiment directories.
start_index = 2700
unique_n_vals = [100,200,300,400]
repetitions = 4
experiment_caption_prefix = '' # Prefix for legend labels, e.g., "n=1000"
USE_LAST_CHECKPOINT = False # Set to True to use only last.ckpt

# Generate the experiment directories dynamically
experiment_dirs = generate_experiment_dir_strings(start_index, unique_n_vals, repetitions)
print(f"Generated experiment directories: {experiment_dirs}") # Optional: uncomment to verify

# Calculate the end index for filename generation
end_index = start_index + len(unique_n_vals) * repetitions - 1
base_roc_filename = f'classifiers/roc_curves_{start_index}-{end_index}'
if USE_LAST_CHECKPOINT:
    roc_filename = f'{base_roc_filename}_last.pkl'
else:
    roc_filename = f'{base_roc_filename}.pkl'

calculate_roc_curves(experiment_dirs, experiment_caption_prefix, unique_n_vals, repetitions, use_last=USE_LAST_CHECKPOINT)

rocs = load_roc_curves(roc_filename) # Use the dynamic filename, potentially with _last suffix

if rocs: # Only proceed if loading was successful
    colors = ['#FF0000', '#80FF00', '#00FFFF', '#7F00FF']
    # Example: Select the best checkpoint (top_n_per_rep=1) from each repetition
    plot_roc_curves_with_bounds(rocs, colors, repetitions=repetitions, top_n_per_rep=1, legend_title='Training Set Size (Best Checkpoint per Rep)')
    plt.show()
else:
    print(f"Could not load ROC data from {roc_filename}. Skipping plot.")