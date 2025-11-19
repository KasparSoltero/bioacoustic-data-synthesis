#this script is used to evaluate the performance of previous experiments

#!/usr/bin/env python3

import argparse
import yaml
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import base64
import pandas as pd

# --- YAML Custom Constructors for NumPy types ---
def numpy_ndarray_name_constructor(loader, node):
    return np.ndarray

def python_tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node, deep=True))

def numpy_dtype_constructor(loader, node):
    value = loader.construct_mapping(node, deep=True)

    if not isinstance(value, dict) or 'args' not in value or not isinstance(value['args'], list) or not value['args']:
        raise yaml.YAMLError(f"Unexpected structure for numpy.dtype args mapping: {value}")

    dtype_arg = value['args'][0]

    if not isinstance(dtype_arg, str):
         try:
             dtype_arg = str(dtype_arg)
         except Exception as e:
             raise yaml.YAMLError(f"Could not interpret dtype argument '{dtype_arg}' (type {type(dtype_arg)}) as string: {e}")

    try:
        return np.dtype(dtype_arg)
    except Exception as e:
        raise yaml.YAMLError(f"Failed to create numpy.dtype from argument '{dtype_arg}': {e}")


def numpy_scalar_constructor(loader, node):
    value = loader.construct_sequence(node, deep=True)

    if not isinstance(value, list) or len(value) != 2:
        raise yaml.YAMLError(f"Unexpected structure for numpy.scalar sequence value: {value}")

    np_dtype = value[0]
    binary_data = value[1]

    if not isinstance(np_dtype, np.dtype):
        raise yaml.YAMLError(f"Expected numpy.dtype object as first item in scalar constructor, but got {type(np_dtype)}")
    if not isinstance(binary_data, bytes): # Expect bytes due to !!binary tag
         raise yaml.YAMLError(f"Expected binary data (bytes) as second item in scalar constructor, but got {type(binary_data)}")

    try:
        scalar_array = np.frombuffer(binary_data, dtype=np_dtype)

        if scalar_array.size == 1:
            return float(scalar_array.item())
        else:
             print(f"Warning: numpy.scalar constructor encountered non-scalar data (size {scalar_array.size}). Returning first element as float.")
             if scalar_array.size > 0:
                 return float(scalar_array[0])
             else:
                 print(f"Warning: numpy.scalar constructor encountered empty array (size {scalar_array.size}). Returning NaN.")
                 return float('nan')

    except Exception as e:
        raise yaml.YAMLError(f"Error processing numpy.scalar binary data: {e}")


def numpy_ndarray_constructor(loader, node):
    value = loader.construct_mapping(node, deep=True)

    if not isinstance(value, dict) or 'args' not in value or 'state' not in value:
         raise yaml.YAMLError(f"Unexpected structure for numpy._reconstruct mapping value: {value}")

    state = value['state']

    if not isinstance(state, tuple) or len(state) != 5:
         raise yaml.YAMLError(f"Unexpected state tuple structure for numpy._reconstruct state value: {state}")

    # shape_from_state = state[1]
    np_dtype = state[2]
    binary_data = state[4] # Expect bytes due to !!binary tag

    if not isinstance(np_dtype, np.dtype):
         raise yaml.YAMLError(f"Expected numpy.dtype object in state tuple for _reconstruct, but got {type(np_dtype)}")
    if not isinstance(binary_data, bytes): # Expect bytes due to !!binary tag
         raise yaml.YAMLError(f"Expected binary data (bytes) in state tuple for _reconstruct, but got {type(binary_data)}")

    try:
        array = np.frombuffer(binary_data, dtype=np_dtype)

        if array.size == 1:
             return float(array.item())
        else:
            print(f"Warning: YAML _reconstruct constructor encountered multi-element array (shape {array.shape}) in metric. Using first element.")
            if array.size > 0:
                 return float(array[0])
            else:
                 print(f"Warning: YAML _reconstruct constructor encountered empty array (shape {array.shape}). Returning NaN.")
                 return float('nan')

    except Exception as e:
        raise yaml.YAMLError(f"Error processing numpy._reconstruct binary data: {e}")


class CustomYamlLoader(yaml.FullLoader):
    pass

CustomYamlLoader.add_constructor('tag:yaml.org,2002:python/name:numpy.ndarray', numpy_ndarray_name_constructor)
CustomYamlLoader.add_constructor('tag:yaml.org,2002:python/tuple', python_tuple_constructor)
CustomYamlLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.dtype', numpy_dtype_constructor)
CustomYamlLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy._core.multiarray.scalar', numpy_scalar_constructor)
CustomYamlLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy._core.multiarray._reconstruct', numpy_ndarray_constructor)
# --- End YAML Custom Constructors ---


def load_metrics(metrics_file_path, exp_id_for_warning):
    if not metrics_file_path.exists():
        print(f"Warning: Metrics file not found: {metrics_file_path}")
        return None
    try:
        with open(metrics_file_path, 'r') as f:
            metrics_data_raw = yaml.load(f, Loader=CustomYamlLoader)

        if metrics_data_raw is None:
            print(f"Warning: Metrics file {metrics_file_path} is empty.")
            return None

        raw_epoch_list = []
        if isinstance(metrics_data_raw, dict):
            raw_epoch_list = [metrics_data_raw]
        elif isinstance(metrics_data_raw, list):
            raw_epoch_list = metrics_data_raw
        else:
            print(f"Warning: Unexpected top-level structure in {metrics_file_path}: {type(metrics_data_raw)}. Expected list or dict.")
            return None

        processed_metrics = []
        for i, epoch_data in enumerate(raw_epoch_list):
            if not isinstance(epoch_data, dict):
                print(f"Warning: Expected a dictionary for epoch data in {metrics_file_path} (entry {i}), but got {type(epoch_data)}. Skipping this entry.")
                continue

            processed_epoch = {}
            for key, value in epoch_data.items():
                # For known numerical metrics, ensure they are floats or NaN
                if key in ['train_loss', 'val_loss', 'rl-loss', 'val_auc', 'rl-auc', 'val_f1', 'rl-f1', 'val_precision', 'rl-precision', 'val_recall', 'rl-recall']:
                    if isinstance(value, (int, float)):
                        processed_epoch[key] = float(value)
                    elif isinstance(value, np.generic):
                         processed_epoch[key] = float(value)
                    elif isinstance(value, np.ndarray) and (value.ndim == 0 or value.size == 1):
                         processed_epoch[key] = float(value.item())
                    else:
                         print(f"Warning: Metric '{key}' in ExpID {exp_id_for_warning} has non-numeric value ('{value}', type {type(value)}). Setting to NaN.")
                         processed_epoch[key] = float('nan')
                else:
                     # Keep other values as is, or convert to string defensively
                     try:
                          processed_epoch[key] = str(value)
                     except Exception:
                          processed_epoch[key] = repr(value) # Fallback


            processed_metrics.append(processed_epoch)

        if not processed_metrics and raw_epoch_list:
             print(f"Warning: No valid epoch dictionaries processed from {metrics_file_path}.")
             return None


        return processed_metrics

    except yaml.YAMLError as e:
        print(f"YAML parsing error in {metrics_file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading or processing metrics from {metrics_file_path}: {e}")
        return None


def display_metrics_terminal(exp_id, n_val, metrics_data):
    print(f"\n--- Experiment ID: {exp_id} (n_val: {n_val}) ---")
    if not metrics_data or not isinstance(metrics_data, list) or not metrics_data[0] or not isinstance(metrics_data[0], dict):
        print("No metrics data available or data is malformed.")
        return

    sample_epoch_keys = metrics_data[0].keys()
    header = ["Epoch"] + sorted([k for k in sample_epoch_keys if k.lower() != 'epoch'])

    col_widths = {h: len(h) for h in header}
    for i, epoch_metrics in enumerate(metrics_data):
        col_widths["Epoch"] = max(col_widths["Epoch"], len(str(i)))
        for key in header[1:]:
            value = epoch_metrics.get(key)
            if isinstance(value, float) and not np.isnan(value):
                 val_str = f"{value:.4f}"
            elif value is None:
                 val_str = 'N/A'
            else:
                 try:
                      val_str = str(value)
                 except Exception:
                      val_str = repr(value)
            col_widths[key] = max(col_widths[key], len(val_str))

    col_widths = {k: w + 2 for k, w in col_widths.items()}

    header_line_formatted = "|".join(f" {h:<{col_widths[h]-1}}" for h in header)
    separator_line = "+".join("-" * col_widths[h] for h in header)

    print(f"+{separator_line}+")
    print(f"|{header_line_formatted}|")
    print(f"+{separator_line}+")

    for i, epoch_metrics in enumerate(metrics_data):
        row_values = [f" {i:<{col_widths['Epoch']-1}}"]
        for key in header[1:]:
            value = epoch_metrics.get(key)
            if isinstance(value, float):
                if np.isnan(value):
                    val_str = 'NaN'
                else:
                    val_str = f"{value:.4f}"
            elif value is None:
                 val_str = 'N/A'
            else:
                try:
                    val_str = str(value)
                except Exception:
                    val_str = repr(value)

            row_values.append(f" {val_str:<{col_widths[key]-1}}")
        print("|" + "|".join(row_values) + "|")

    print(f"+{separator_line}+")


# plot_metrics function updated
def plot_metrics(exp_id, n_val, metrics_data):
    if not metrics_data or not isinstance(metrics_data, list) or not metrics_data[0] or not isinstance(metrics_data[0], dict):
        print(f"No metrics data to plot for Experiment ID: {exp_id} (n_val: {n_val})")
        return

    epochs = list(range(len(metrics_data)))

    # Safely get metric values, defaulting to NaN if key is missing or value is non-numeric
    def get_metric_values(metric_key):
        values = []
        for m in metrics_data:
            value = m.get(metric_key)
            if isinstance(value, (int, float)):
                values.append(float(value)) # Ensure float
            else:
                values.append(float('nan')) # Use NaN for missing or non-numeric

        # Filter out leading NaNs if the metric doesn't start from epoch 0
        # Find the first non-NaN index
        first_valid_idx = np.where(~np.isnan(values))[0]
        if first_valid_idx.size > 0:
             first_valid_idx = first_valid_idx[0]
             # If first valid index is > 0, plot from there
             return np.array(values), np.array(epochs), first_valid_idx
        else:
             # All values are NaN
             return np.array(values), np.array(epochs), 0 # Indicate no valid data

    # Get data for plotting
    val_auc, epochs_val_auc, first_valid_idx_val_auc = get_metric_values('val_auc')
    val_f1, _, first_valid_idx_val_f1 = get_metric_values('val_f1')
    val_precision, _, first_valid_idx_val_precision = get_metric_values('val_precision')
    val_recall, _, first_valid_idx_val_recall = get_metric_values('val_recall')

    train_loss, epochs_train_loss, first_valid_idx_train_loss = get_metric_values('train_loss')
    val_loss, _, first_valid_idx_val_loss = get_metric_values('val_loss')

    rl_loss, epochs_rl_loss, first_valid_idx_rl_loss = get_metric_values('rl-loss')
    rl_f1, _, first_valid_idx_rl_f1 = get_metric_values('rl-f1')
    rl_auc, _, first_valid_idx_rl_auc = get_metric_values('rl-auc')
    rl_precision, _, first_valid_idx_rl_precision = get_metric_values('rl-precision')
    rl_recall, _, first_valid_idx_rl_recall = get_metric_values('rl-recall')


    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax1, ax2 = axes
    fig.suptitle(f"Metrics for Experiment ID: {exp_id} (n_val: {n_val})", fontsize=16)

    # Plot Validation Metrics on ax1
    # Plot only from the first epoch where data is available for that metric
    if not np.all(np.isnan(val_auc)):
        ax1.plot(epochs_val_auc[first_valid_idx_val_auc:], val_auc[first_valid_idx_val_auc:], label='Validation AUC', marker='o', linestyle='-', color='#f20085')
    if not np.all(np.isnan(val_f1)):
         ax1.plot(epochs[first_valid_idx_val_f1:], val_f1[first_valid_idx_val_f1:], label='Validation F1-Score', marker='s', linestyle='--', color='#0066FF')
    if not np.all(np.isnan(val_precision)):
         ax1.plot(epochs[first_valid_idx_val_precision:], val_precision[first_valid_idx_val_precision:], label='Validation Precision', marker='^', linestyle='-.', color='#ff4800')
    if not np.all(np.isnan(val_recall)):
         ax1.plot(epochs[first_valid_idx_val_recall:], val_recall[first_valid_idx_val_recall:], label='Validation Recall', marker='d', linestyle=':', color='#00ba5d')
    if not np.all(np.isnan(train_loss)):
        ax1.plot(epochs_train_loss[first_valid_idx_train_loss:], train_loss[first_valid_idx_train_loss:], label='Train Loss', marker='.', linestyle='-', color='#f7df00')
    if not np.all(np.isnan(val_loss)):
        ax1.plot(epochs[first_valid_idx_val_loss:], val_loss[first_valid_idx_val_loss:], label='Validation Loss', marker='.', linestyle='--', color='#7F00FF')
    

    ax1.set_xlabel('Epoch', fontsize='16')
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_ylabel('Value', fontsize='16')
    ax1.legend(framealpha=1.0,    # Non-transparent background
           fancybox=False,    # Square edges instead of rounded
           fontsize=15,       # Larger font size (adjust as needed)
           edgecolor='black') # Add border color
    ax1.set_ylim(0, 1) # Set y-axis limit for metrics
    ax1.set_xlim(0,15) # temp - remove
    ax1.grid(True)

    # Plot Loss and RL Metrics on ax2
    if not np.all(np.isnan(rl_auc)):
         ax2.plot(epochs[first_valid_idx_rl_auc:], rl_auc[first_valid_idx_rl_auc:], label='RW AUC', marker='o', linestyle='-', color='#f20085')
    if not np.all(np.isnan(rl_f1)):
         ax2.plot(epochs[first_valid_idx_rl_f1:], rl_f1[first_valid_idx_rl_f1:], label='RW F1-Score', marker='s', linestyle='--', color='#0066FF')
    if not np.all(np.isnan(rl_precision)):
         ax2.plot(epochs[first_valid_idx_rl_precision:], rl_precision[first_valid_idx_rl_precision:], label='RW Precision', marker='^', linestyle='-.', color='#ff4800')
    if not np.all(np.isnan(rl_recall)):
         ax2.plot(epochs[first_valid_idx_rl_recall:], rl_recall[first_valid_idx_rl_recall:], label='RW Recall', marker='d', linestyle=':', color='#00ba5d')
    # if not np.all(np.isnan(rl_loss)):
    #     ax2.plot(epochs_rl_loss[first_valid_idx_rl_loss:], rl_loss[first_valid_idx_rl_loss:], label='RW Loss', marker='.', linestyle='-', color='#ccc')

    ax2.set_xlabel('Epoch', fontsize='16')
    ax2.set_ylabel('Value', fontsize='16') # Can be Loss or Metric Value
    ax2.tick_params(axis='both', labelsize=16)
    ax2.legend(framealpha=1.0,    # Non-transparent background
           fancybox=False,    # Square edges instead of rounded
           fontsize=15,       # Larger font size (adjust as needed)
           edgecolor='black') # Add border color
    ax2.set_ylim(0, 1) # Set y-axis limit for metrics
    ax2.set_xlim(0,15) # temp - remove
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def display_summary_table(all_max_metrics_data, unique_n_vals_sorted, output_csv_path=None):
    """
    Displays a summary table of mean metrics across repetitions for each n_val
    and writes it to a CSV file if output_csv_path is provided.
    Mean is the average of values (e.g., max_auc, or metric_at_max_auc).
    Deviation is the sample standard deviation (ddof=1) across repetitions.
    """
    summary_by_nval = {} # Key: n_val, Value: dict of {'metric_name': [list of vals_from_reps]}

    # Assuming metrics_to_summarize are the keys like 'max_rl-auc' etc. from your collection logic
    # Or, if your collection logic still populates with 'rl-auc', adjust here or there.
    # Let's assume all_max_metrics_data has keys like 'max_rl-auc'
    
    # Infer metrics_to_summarize from the data if not fixed, or use your fixed list
    # This example assumes the keys in all_max_metrics_data are like 'max_METRICNAME'
    # and we want to report on 'METRICNAME'
    
    # Based on your previous code, the keys in all_max_metrics_data will be 'max_rl-auc', etc.
    # and your metrics_to_summarize in main was ['rl-auc', 'rl-f1', 'rl-recall', 'rl-precision']
    # So we will use the original `metrics_to_summarize` and append "max_" to access data.
    
    metrics_to_report = ['rl-auc', 'rl-f1', 'rl-recall', 'rl-precision'] # The base metric names

    for record in all_max_metrics_data:
        n_val = record['n_val']
        if n_val not in summary_by_nval:
            summary_by_nval[n_val] = {metric: [] for metric in metrics_to_report}

        for metric_base_name in metrics_to_report:
            # The keys in record are like 'max_rl-auc'
            metric_key_in_record = f'max_{metric_base_name}' 
            value_from_record = record.get(metric_key_in_record)
            if value_from_record is not None and not np.isnan(value_from_record):
                summary_by_nval[n_val][metric_base_name].append(value_from_record)

    processed_summary_rows = []
    for n_val in unique_n_vals_sorted: # Iterate in sorted order of n_val
        row_data = {'n_val': n_val}
        
        if n_val in summary_by_nval:
            for metric_base_name in metrics_to_report:
                values = summary_by_nval[n_val].get(metric_base_name, [])
                if values: # If there are valid values for this metric and n_val
                    mean_val = np.mean(values)
                    # Calculate sample standard deviation.
                    # np.std returns NaN if len(values) < 2 and ddof=1.
                    std_dev = np.std(values, ddof=1) 
                    
                    row_data[f'mean_{metric_base_name}'] = mean_val
                    row_data[f'dev_{metric_base_name}'] = std_dev
                else: # No valid values found (e.g., all NaNs or metric missing)
                    row_data[f'mean_{metric_base_name}'] = np.nan
                    row_data[f'dev_{metric_base_name}'] = np.nan
        else: # This n_val might not have had any successful runs or data
            for metric_base_name in metrics_to_report:
                row_data[f'mean_{metric_base_name}'] = np.nan
                row_data[f'dev_{metric_base_name}'] = np.nan
        
        processed_summary_rows.append(row_data)

    # --- Print the table ---
    print("\n\n--- Overall Summary Across Repetitions (Mean ± Std. Dev.) ---")
    
    headers = ["n_val"] + [f"mean {m} (±std)" for m in metrics_to_report]
    col_widths = {h: len(h) for h in headers}

    # Calculate column widths
    for r_data in processed_summary_rows:
        col_widths["n_val"] = max(col_widths["n_val"], len(str(r_data['n_val'])))
        for metric_base_name in metrics_to_report:
            header_key = f"mean {metric_base_name} (±std)" # Updated header text
            mean_val = r_data.get(f'mean_{metric_base_name}')
            dev_val = r_data.get(f'dev_{metric_base_name}') # This is now std_dev
            
            if pd.isna(mean_val) or pd.isna(dev_val): 
                val_str = "N/A"
            else:
                val_str = f"{mean_val:.4f} ± {dev_val:.4f}"
            col_widths[header_key] = max(col_widths[header_key], len(val_str))
    
    # Add padding to column widths
    col_widths = {k: w + 2 for k, w in col_widths.items()}

    # Print table header
    header_line_formatted = "|".join(f" {h:<{col_widths[h]-1}}" for h in headers)
    separator_line = "+".join("-" * col_widths[h] for h in headers)

    print(f"+{separator_line}+")
    print(f"|{header_line_formatted}|")
    print(f"+{separator_line}+")

    # Print table rows
    for r_data in processed_summary_rows:
        row_values_str = [f" {str(r_data['n_val']):<{col_widths['n_val']-1}}"]
        for metric_base_name in metrics_to_report:
            header_key = f"mean {metric_base_name} (±std)" # Updated header text
            mean_val = r_data.get(f'mean_{metric_base_name}')
            dev_val = r_data.get(f'dev_{metric_base_name}') # This is now std_dev
            
            if pd.isna(mean_val) or pd.isna(dev_val):
                val_str = "N/A"
            else:
                val_str = f"{mean_val:.4f} ± {dev_val:.4f}"
            row_values_str.append(f" {val_str:<{col_widths[header_key]-1}}")
        print("|" + "|".join(row_values_str) + "|")
    
    print(f"+{separator_line}+")

    print(f"+{separator_line}+")

    if output_csv_path:
        try:
            # Create a DataFrame from processed_summary_rows for CSV export
            # We need to re-format the data slightly for a clean CSV
            csv_rows = []
            for r_data in processed_summary_rows:
                csv_row = {'n_val': r_data['n_val']}
                for metric_base_name in metrics_to_report:
                    mean_val = r_data.get(f'mean_{metric_base_name}')
                    dev_val = r_data.get(f'dev_{metric_base_name}')
                    csv_row[f'mean_{metric_base_name}'] = mean_val if not pd.isna(mean_val) else None
                    csv_row[f'std_dev_{metric_base_name}'] = dev_val if not pd.isna(dev_val) else None # std_dev is more standard than 'dev'
                csv_rows.append(csv_row)
            
            summary_df = pd.DataFrame(csv_rows)
            
            # Define column order for the CSV
            csv_columns = ['n_val']
            for metric_base_name in metrics_to_report:
                csv_columns.append(f'mean_{metric_base_name}')
                csv_columns.append(f'std_dev_{metric_base_name}')
            
            summary_df = summary_df[csv_columns] # Ensure correct column order
            
            output_csv_path = Path(output_csv_path) # Ensure it's a Path object
            output_csv_path.parent.mkdir(parents=True, exist_ok=True) # Create parent dirs if they don't exist
            summary_df.to_csv(output_csv_path, index=False, float_format='%.4f')
            print(f"\nSummary table also saved to: {output_csv_path}")
        except Exception as e:
            print(f"\nError saving summary table to CSV {output_csv_path}: {e}")
    
    return processed_summary_rows, metrics_to_report


def plot_summary_metrics(processed_summary_data, metrics_to_report, unique_n_vals_sorted):
    """
    Plots mean metrics from the summary table against n_val.
    """
    if not processed_summary_data:
        print("No summary data available to plot.")
        return

    plt.figure(figsize=(12, 8))
    
    styles = {
        'rl-auc': {'marker': 'o', 'linestyle': '-', 'color': '#f20085', 'label': 'Mean RL AUC'},
        'rl-f1': {'marker': 's', 'linestyle': '--', 'color': '#0066FF', 'label': 'Mean RL F1-Score'},
        'rl-precision': {'marker': '^', 'linestyle': '-.', 'color': '#ff4800', 'label': 'Mean RL Precision'},
        'rl-recall': {'marker': 'd', 'linestyle': ':', 'color': '#00ba5d', 'label': 'Mean RL Recall'}
    }

    # Collect data for plotting, ensuring n_vals are aligned for each metric series
    plot_data_collections = {metric: {'x': [], 'y': [], 'std': []} for metric in metrics_to_report if metric in styles}

    for n_val_current in unique_n_vals_sorted: # Iterate over all defined n_vals to ensure consistent x-axis consideration
        data_for_nval_dict = next((item for item in processed_summary_data if item['n_val'] == n_val_current), None)
        
        if data_for_nval_dict: # If there's summary data for this n_val
            for metric_base_name in metrics_to_report:
                if metric_base_name in styles: # Process only metrics we have styles for
                    mean_val = data_for_nval_dict.get(f'mean_{metric_base_name}')
                    dev_val = data_for_nval_dict.get(f'dev_{metric_base_name}') # Get the standard deviation

                    if mean_val is not None and not pd.isna(mean_val):
                        plot_data_collections[metric_base_name]['x'].append(n_val_current)
                        plot_data_collections[metric_base_name]['y'].append(mean_val)
                        # Also store std dev, use np.nan if missing or NaN to align with mean
                        if dev_val is not None and not pd.isna(dev_val):
                            plot_data_collections[metric_base_name]['std'].append(dev_val)
                        else:
                            plot_data_collections[metric_base_name]['std'].append(np.nan)
                    # else: # If mean_val is NaN or missing
                        # To maintain alignment if we were to plot NaNs for means:
                        # plot_data_collections[metric_base_name]['x'].append(n_val_current)
                        # plot_data_collections[metric_base_name]['y'].append(np.nan)
                        # plot_data_collections[metric_base_name]['std'].append(np.nan)
        # else: # No summary data at all for this n_val_current across all metrics
            # If we wanted to ensure all lines show a gap at this n_val:
            # for metric_base_name in metrics_to_report:
            #     if metric_base_name in styles:
            #         plot_data_collections[metric_base_name]['x'].append(n_val_current)
            #         plot_data_collections[metric_base_name]['y'].append(np.nan)


    min_plot_value = float('inf')

    for metric_base_name, data_points in plot_data_collections.items():
        if data_points['x']:  # Only plot if there's data for this metric
            style = styles[metric_base_name]
            x_coords = np.array(data_points['x'])
            y_coords = np.array(data_points['y'])
            std_coords = np.array(data_points['std'])

            # Plot the mean line
            plt.plot(x_coords, y_coords,
                     marker=style['marker'], linestyle=style['linestyle'], color=style['color'], label=style['label'])

            # Add transparent bounds for std dev if std_coords are available and not all NaN
            if len(std_coords) == len(y_coords) and not np.all(np.isnan(std_coords)):
                upper_bound = y_coords + std_coords
                lower_bound = y_coords - std_coords
                plt.fill_between(x_coords, lower_bound, upper_bound, color=style['color'], alpha=0.2)
                
                # Update min_plot_value, considering only valid (non-NaN) lower bounds
                current_min_lower_bound = np.nanmin(lower_bound)
                if not np.isnan(current_min_lower_bound):
                    min_plot_value = min(min_plot_value, current_min_lower_bound)
            else: # If no std_dev, consider the y_coords themselves for min_plot_value
                current_min_y = np.nanmin(y_coords)
                if not np.isnan(current_min_y):
                    min_plot_value = min(min_plot_value, current_min_y)

    plt.xlabel('n_val (Number of Validation Samples per Class)')
    plt.ylabel('Mean Metric Value')
    plt.title('Summary of Mean RL Metrics vs. n_val')
    
    if unique_n_vals_sorted:
        plt.xticks(ticks=unique_n_vals_sorted, labels=[str(n) for n in unique_n_vals_sorted])
        # Consider adjusting xlim if n_vals are sparse or to add padding
        # if len(unique_n_vals_sorted) > 1:
        #     plt.xlim(min(unique_n_vals_sorted) - (unique_n_vals_sorted[1]-unique_n_vals_sorted[0])*0.1 if len(unique_n_vals_sorted)>1 else unique_n_vals_sorted[0]-0.5 ,
        #              max(unique_n_vals_sorted) + (unique_n_vals_sorted[1]-unique_n_vals_sorted[0])*0.1 if len(unique_n_vals_sorted)>1 else unique_n_vals_sorted[0]+0.5)

    plt.legend(loc='best')
    plt.grid(True)
    
    # Determine y-axis limits
    # If min_plot_value remained inf, it means no valid data points were found. Default to 0.
    lower_y_lim = min_plot_value if min_plot_value != float('inf') else 0.0
    # Add a small padding below the minimum, but don't go below a sensible floor like -0.05.
    # Ensure the padding doesn't make the lower limit too far from the actual data if data is close to 0.
    padding = 0.05
    effective_lower_y_lim = lower_y_lim - padding
    
    # Ensure the lower limit is not excessively low if all data is positive and close to zero.
    # And also ensure it's not above 0 if min_plot_value was actually negative.
    if lower_y_lim >= 0: # if all data points (mean-std) are non-negative
        final_lower_y_lim = max(-padding, effective_lower_y_lim) # Don't go below -0.05, but allow slight negative for padding
    else: # if some data points (mean-std) are negative
        final_lower_y_lim = effective_lower_y_lim

    plt.ylim(final_lower_y_lim, 1.0)
    plt.xlim(500,8000)
    
    plt.tight_layout()
    plt.show()

def plot_rl_loss(n_vals_to_plot, experiment_details, base_results_path, use_lowest_loss=False):
    """
    Plot RL-loss curves over epochs for specified n values.
    
    Args:
        n_vals_to_plot: List of n values to include in the plot
        experiment_details: List of experiment details (dicts with exp_id, n_val, repetition_num)
        base_results_path: Path object pointing to the base directory for results
        use_lowest_loss: If True, only plot the lowest loss curve for each n, otherwise plot mean±std
    """
    if not n_vals_to_plot:
        print("No n values provided to plot RL-loss curves.")
        return
    
    # Filter experiments to only include those with the requested n values
    filtered_experiments = [exp for exp in experiment_details if exp['n_val'] in n_vals_to_plot]
    
    if not filtered_experiments:
        print(f"No experiments found for n values: {n_vals_to_plot}")
        return
    
    # Group experiments by n_val
    experiments_by_n = {}
    for exp in filtered_experiments:
        n_val = exp['n_val']
        if n_val not in experiments_by_n:
            experiments_by_n[n_val] = []
        experiments_by_n[n_val].append(exp)
    
    plt.figure(figsize=(12, 8))
    
    # Colors for different n values
    colors = ['#f20085', '#0066FF', '#ff4800', '#00ba5d', '#7F00FF', '#f7df00']
    min_epoch = 9999
    
    for i, (n_val, exps) in enumerate(sorted(experiments_by_n.items())):
        color = colors[i % len(colors)]
        
        # Collect all RL-loss data for this n value
        all_rl_loss_data = []
        exp_ids = []  # Keep track of which exp_id corresponds to each loss curve
        max_epochs = 0
        
        for exp in exps:
            exp_id = exp['exp_id']
            exp_dir_name = f'Exp_{exp_id}'
            metrics_file_name = f'metrics_{exp_id}.yaml'
            metrics_file_path = base_results_path / exp_dir_name / metrics_file_name
            
            metrics_data = load_metrics(metrics_file_path, exp_id)
            
            if metrics_data:
                # Extract RL-loss values from all epochs
                rl_loss_values = []
                for epoch_data in metrics_data:
                    rl_loss = epoch_data.get('rl-loss')
                    if isinstance(rl_loss, (int, float)) and not np.isnan(rl_loss):
                        rl_loss_values.append(float(rl_loss))
                    else:
                        rl_loss_values.append(np.nan)
                
                if rl_loss_values and not all(np.isnan(x) for x in rl_loss_values):
                    all_rl_loss_data.append(rl_loss_values)
                    exp_ids.append(exp_id)
                    max_epochs = max(max_epochs, len(rl_loss_values))
        
        if not all_rl_loss_data:
            print(f"No valid RL-loss data found for n_val = {n_val}")
            continue
        
        # Pad shorter arrays with NaN to ensure equal length
        for i in range(len(all_rl_loss_data)):
            if len(all_rl_loss_data[i]) < max_epochs:
                all_rl_loss_data[i].extend([np.nan] * (max_epochs - len(all_rl_loss_data[i])))
        
        # Convert to numpy array for easier calculation
        all_rl_loss_array = np.array(all_rl_loss_data)
        
        if use_lowest_loss:
            # Find the experiment with the lowest mean loss (ignoring NaNs)
            mean_losses = np.nanmean(all_rl_loss_array, axis=1)
            if not np.all(np.isnan(mean_losses)):
                lowest_idx = np.nanargmin(mean_losses)
                lowest_loss_curve = all_rl_loss_array[lowest_idx]
                lowest_exp_id = exp_ids[lowest_idx]
                
                # Create x-axis (epochs)
                epochs = np.arange(len(lowest_loss_curve))
                if len(lowest_loss_curve)<min_epoch:
                    min_epoch = len(lowest_loss_curve)-1
                
                # Plot only the lowest loss curve
                plt.plot(epochs, lowest_loss_curve, 
                         label=f'n = {n_val} (lowest: Exp_{lowest_exp_id})', 
                         color=color, linewidth=2)
            else:
                print(f"All loss values are NaN for n_val = {n_val}")
        else:
            # Calculate mean and std for each epoch
            mean_rl_loss = np.nanmean(all_rl_loss_array, axis=0)
            std_rl_loss = np.nanstd(all_rl_loss_array, axis=0)
            
            # Create x-axis (epochs)
            epochs = np.arange(len(mean_rl_loss))
            if len(mean_rl_loss)<min_epoch:
                min_epoch = len(mean_rl_loss)-1
            
            # Plot mean line
            plt.plot(epochs, mean_rl_loss, label=f'n = {n_val}', color=color, linewidth=2)
            
            # Add transparent bounds for std deviation
            plt.fill_between(
                epochs,
                mean_rl_loss - std_rl_loss,
                mean_rl_loss + std_rl_loss,
                color=color,
                alpha=0.2
            )
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RL-Loss', fontsize=12)
    plt.xlim(0,min_epoch)
    plt.ylim(0,1.5)
    title_suffix = "Lowest Loss Curves" if use_lowest_loss else "Mean ± Std Dev"
    plt.title(f'RL-Loss Across Epochs for Different n Values ({title_suffix})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate and display metrics from experiment YAML files.")
    parser.add_argument('--start_index', type=int, required=True, help="Starting index for experiment names (e.g., 2600).")
    parser.add_argument('--n_vals', type=str, required=True, help="Comma-separated list of unique 'n' values (e.g., '1,2,3,5,10').")
    parser.add_argument('--repetitions', type=int, required=True, help="Number of repetitions for each 'n' value.")
    parser.add_argument('--plot_individual', action='store_true', help="If set, display plots for each individual experiment's metrics.")
    parser.add_argument('--plot_summary', action='store_true', help="If set, display a summary plot of mean metrics vs n_val after the summary table.")
    parser.add_argument('--plot_rl_loss', type=str, help="Comma-separated list of n values to plot RL-loss curves for.")
    parser.add_argument('--use_lowest_loss', action='store_true', help="If set with --plot_rl_loss, only show the lowest loss curve for each n.")
    parser.add_argument('--output_csv_file', type=str, default=False, help="Name for the output CSV file for the summary table.")
    args = parser.parse_args()

    try:
        str_n_vals = [n.strip() for n in args.n_vals.split(',')]
        unique_n_vals = []
        for s_val in str_n_vals:
            try:
                unique_n_vals.append(int(s_val))
            except ValueError:
                 try:
                     unique_n_vals.append(float(s_val))
                 except ValueError:
                     print(f"Error: Could not convert '{s_val}' in --n_vals to a number. Skipping this value.")
        
        if not unique_n_vals:
            print("Error: --n_vals resulted in an empty list of valid numbers or all conversions failed.")
            return
        
        unique_n_vals = sorted(list(set(unique_n_vals))) # Ensure unique and sorted

    except Exception as e:
        print(f"An unexpected error occurred while parsing --n_vals: {e}")
        return


    experiment_details_to_process = []
    num_unique_vals = len(unique_n_vals)

    for val_idx, n_val_current in enumerate(unique_n_vals): # Use sorted unique_n_vals
        for rep_idx in range(args.repetitions):
            # The formula for exp_id might need adjustment if n_vals are not contiguous or simple
            # Assuming start_index is the base for the very first experiment overall.
            # Original logic: exp_id_for_file_and_dir = args.start_index + (rep_idx * num_unique_vals) + val_idx
            # This implies a specific ordering of experiments that must match how they were run/named.
            exp_id_for_file_and_dir = args.start_index + (rep_idx * num_unique_vals) + unique_n_vals.index(n_val_current)

            experiment_details_to_process.append({
                'exp_id': exp_id_for_file_and_dir,
                'n_val': n_val_current,
                'repetition_num': rep_idx,
            })

    base_results_path = Path('classifiers/evaluation_results')
    
    all_experiments_max_metrics = [] # To store max metrics for the summary table

    metrics_to_summarize = ['rl-auc', 'rl-f1', 'rl-recall', 'rl-precision']

    # Process the plot_rl_loss argument if provided
    if args.plot_rl_loss:
        try:
            n_vals_to_plot = []
            for n in args.plot_rl_loss.split(','):
                try:
                    n_vals_to_plot.append(int(n.strip()))
                except ValueError:
                    try:
                        n_vals_to_plot.append(float(n.strip()))
                    except ValueError:
                        print(f"Warning: Could not convert '{n.strip()}' in --plot_rl_loss to a number. Skipping.")
            
            if n_vals_to_plot:
                plot_rl_loss(
                    n_vals_to_plot, 
                    experiment_details_to_process, 
                    base_results_path, 
                    use_lowest_loss=args.use_lowest_loss
                )
            else:
                print("No valid n values provided for --plot_rl_loss")
        except Exception as e:
            print(f"Error processing --plot_rl_loss argument: {e}")

    for exp_info in experiment_details_to_process:
        exp_id = exp_info['exp_id']
        n_val = exp_info['n_val'] # This is the specific n_val for this experiment

        exp_dir_name = f'Exp_{exp_id}'
        metrics_file_name = f'metrics_{exp_id}.yaml'
        metrics_file_path = base_results_path / exp_dir_name / metrics_file_name

        print(f"\nProcessing: {metrics_file_path} (n_val: {n_val}, Rep: {exp_info['repetition_num']})")

        metrics_data = load_metrics(metrics_file_path, exp_id)

        if metrics_data:
            display_metrics_terminal(exp_id, n_val, metrics_data)
            if args.plot_individual:
                plot_metrics(exp_id, n_val, metrics_data)
            
            # --- Collect metrics at the epoch of maximum rl-auc for summary ---
            current_exp_summary_data = {'n_val': n_val, 'repetition_num': exp_info['repetition_num']}
            
            rl_auc_values_with_epoch = []
            # First, gather all rl-auc values with their epoch indices
            for i, epoch_data in enumerate(metrics_data): # metrics_data is list of epoch dicts
                auc_val = epoch_data.get('rl-auc')
                # Ensure auc_val is a valid number before considering it
                if isinstance(auc_val, (float, int)) and not np.isnan(auc_val):
                    rl_auc_values_with_epoch.append({'epoch_idx': i, 'rl-auc': float(auc_val)})
            
            epoch_of_max_rl_auc = -1
            # max_rl_auc_value_at_epoch = -float('inf') # Not strictly needed if we just store the epoch_idx

            if rl_auc_values_with_epoch:
                # Find the entry (and thus epoch_idx) with the maximum rl-auc
                # If multiple epochs have the same max rl-auc, this takes the one with the lowest index (first occurrence)
                best_auc_entry = max(rl_auc_values_with_epoch, key=lambda x: x['rl-auc'])
                epoch_of_max_rl_auc = best_auc_entry['epoch_idx']
            
            if epoch_of_max_rl_auc != -1:
                # We found an epoch with max rl-auc. Get metrics from this specific epoch.
                epoch_data_at_max_auc = metrics_data[epoch_of_max_rl_auc]
                for metric_name in metrics_to_summarize: # ['rl-auc', 'rl-f1', 'rl-recall', 'rl-precision']
                    val = epoch_data_at_max_auc.get(metric_name)
                    if isinstance(val, (float, int)): # This will handle actual numbers; np.nan is a float
                        current_exp_summary_data[f'max_{metric_name}'] = float(val) # Re-using 'max_' prefix for convenience
                    else: # Handles None or other non-numeric types by setting to NaN
                        current_exp_summary_data[f'max_{metric_name}'] = np.nan
            else:
                # No valid rl-auc found for this experiment, or all were NaN.
                # So, we can't determine a "best" epoch based on rl-auc.
                # Set all metrics for this repetition to NaN.
                for metric_name in metrics_to_summarize:
                    current_exp_summary_data[f'max_{metric_name}'] = np.nan
            
            all_experiments_max_metrics.append(current_exp_summary_data)
            # --- End collection for summary ---

        else:
            print(f"Skipping display/plot for Exp {exp_id} (n_val: {n_val}) due to missing, unreadable, or empty/malformed metrics file.")

    # --- After processing all experiments, display the summary table ---
    if all_experiments_max_metrics:
        # Construct the full path for the CSV file
        if args.output_csv_file: csv_output_file_path = base_results_path / args.output_csv_file
        else: csv_output_file_path = False # No CSV output requested
        
        processed_summary_data, reported_metrics = display_summary_table(
            all_experiments_max_metrics,
            unique_n_vals, # This is unique_n_vals_sorted from earlier in main
            output_csv_path=csv_output_file_path
        )
        
        if args.plot_summary and processed_summary_data: # Check if data exists for plotting
            # unique_n_vals is already sorted and contains all n_vals for the x-axis
            plot_summary_metrics(processed_summary_data, reported_metrics, unique_n_vals)
    else:
        print("\nNo data collected from any experiment to generate a summary table.")

if __name__ == '__main__':
    main()