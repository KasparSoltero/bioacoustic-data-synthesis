# visualisation.py
# plotting spectrograms with matplotlib, including log scaling and custom colormaps. 
# Also includes an interactive CLI for quick visualization of generated samples.
import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from spectrogram import Spectrogram

# standard custom colors from previous configuration
colors_dusk = ["#000000", "#2c1044", "#7b1c58", "#c8314e", "#f06f35", "#f5c353", "#ffffff"]
cmap_dusk = LinearSegmentedColormap.from_list("dusk", colors_dusk)
custom_cmaps = {'dusk': cmap_dusk}

def map_freq_to_log_pixels(original_height, freqs, max_freq):
    log_indices = []
    for f in freqs:
        relative_pos = f / max_freq
        log_pos = np.log10(relative_pos * 9 + 1)
        log_index = int(np.round(log_pos * (original_height - 1)))
        log_indices.append(log_index)
    return log_indices

def plot_spectrogram(spec_obj, ax=None, cmap='viridis', db_scale=True, show=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        owns_ax = True
    else:
        owns_ax = False

    vals = spec_obj.values
    if spec_obj.is_complex:
        vals = vals.abs().pow(2)

    vals = vals[0].cpu().numpy()

    if db_scale:
        vals = 10 * np.log10(vals + 1e-10)
        vmin, vmax = np.percentile(vals, [1, 99.9]) # clamp extreme outliers for display
    else:
        vmin, vmax = vals.min(), vals.max()

    duration = spec_obj.waveform.shape[-1] / spec_obj.sample_rate
    nyquist = spec_obj.sample_rate / 2
    height = vals.shape[0]

    if isinstance(cmap, str) and cmap in custom_cmaps:
        cmap = custom_cmaps[cmap]

    im = ax.imshow(vals, aspect='auto', origin='lower', extent=[0, duration, 0, height], cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')

    if spec_obj.is_logscale:
        target_freqs = [0, 1000, 2000, 5000, 10000, int(nyquist)]
        target_freqs = [f for f in target_freqs if f <= nyquist]
        yticks = map_freq_to_log_pixels(height, target_freqs, nyquist)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(f//1000) for f in target_freqs])
    else:
        target_freqs = np.linspace(0, nyquist, 6)
        yticks = np.linspace(0, height, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(f//1000)}" for f in target_freqs])

    if owns_ax and show:
        plt.tight_layout()
        plt.show()

    return im

def interactive_cli():
    print("--- Bioacoustic Spectrogram Visualiser ---")
    path = input("Enter path to audio file: ").strip()
    
    # remove quotes if dragged and dropped into terminal
    path = path.strip("'\"")

    if not os.path.exists(path):
        print("File not found.")
        return

    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    spec = Spectrogram(waveform, sr, power=2.0)
    cmap = 'dusk'

    while True:
        print("\n--- Settings ---")
        print(f"1. Toggle Log Scale (Current: {spec.is_logscale})")
        print(f"2. Toggle Complex/Power (Current Power: {spec.power})")
        print(f"3. Change Colormap (Current: {cmap})")
        print("4. Plot")
        print("5. Quit")
        choice = input("Select option: ").strip()

        if choice == '1':
            if spec.is_logscale:
                spec.to_linear()
            else:
                spec.to_logscale()
        elif choice == '2':
            if spec.is_complex:
                spec.to_real(power=2.0)
            else:
                spec.to_complex()
        elif choice == '3':
            new_cmap = input("Enter colormap name (e.g., dusk, viridis, gray, magma): ").strip()
            if new_cmap:
                cmap = new_cmap
        elif choice == '4':
            print("Rendering plot...")
            plot_spectrogram(spec, cmap=cmap)
        elif choice in ('5', 'q', 'quit'):
            break

if __name__ == "__main__":
    interactive_cli()