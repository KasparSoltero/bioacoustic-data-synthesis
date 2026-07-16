# synthesiser/visualisation.py
# plotting spectrograms with matplotlib, including log scaling and custom colormaps. 
# Also includes an interactive CLI for quick visualization of generated samples.
import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from typing import Optional, Tuple

from synthesiser.spectrogram import Spectrogram

# standard custom colors from previous configuration
colors_dusk = ["#000000", "#2c1044", "#7b1c58", "#c8314e", "#f06f35", "#f5c353", "#ffffff"]
cmap_dusk = LinearSegmentedColormap.from_list("dusk", colors_dusk)
custom_cmaps = {'dusk': cmap_dusk}

def map_freq_to_log_pixels(image_height, freqs, max_freq):
    log_indices = []
    for f in freqs:
        relative_pos = f / max_freq
        log_pos = np.log10(relative_pos * 9 + 1)
        log_indices.append(log_pos * image_height)
    return log_indices

def plot_spectrogram(
    spec_obj,
    ax=None,
    cmap='viridis',
    db_scale=True,
    vmin=None,
    vmax=None,
    show=True,
    annotations=None,
    bandpass_hz: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
):
    """
    Renders a spectrogram object to a matplotlib plot.
    If ax is None, creates a new figure and optionally shows it.
    bandpass_hz: optional (highpass_hz, lowpass_hz) pair — draws horizontal
                 white dotted lines at those frequencies on the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        if title and fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(title)
        owns_ax = True
    else:
        owns_ax = False

    vals = spec_obj.values
    if spec_obj.is_complex:
        vals = vals.abs().pow(2)

    vals = vals[0].cpu().numpy()

    if db_scale:
        vals = 10 * np.log10(vals + 1e-10)
        vmin = -60.0 if vmin is None else vmin
        vmax = 50.0 if vmax is None else vmax
    else:
        vmin = 0.0 if vmin is None else vmin
        vmax = 1.0 if vmax is None else vmax

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
        step = 4000 if nyquist > 12000 else 2000
        target_freqs = np.arange(0, nyquist + 1, step)
        yticks = target_freqs / nyquist * height
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(f//1000)}" for f in target_freqs])

    if annotations:
        total_frames = vals.shape[-1]
        original_height = spec_obj.n_fft // 2 + 1
        
        # Build dynamic colormap for unique classes
        unique_labels = sorted(list(set([ann.get('record').label for ann in annotations if ann.get('record')])))
        num_labels = len(unique_labels)
        color_map = {}
        for i, label in enumerate(unique_labels):
            # hue from 0 to 1, saturation 1, value 1
            hue = i / num_labels if num_labels > 0 else 0
            color_map[label] = tuple(hsv_to_rgb((hue, 1.0, 1.0)))

        for ann in annotations:
            record = ann.get('record')
            label = record.label if record else 'Unknown'
            color = color_map.get(label, (0.0, 1.0, 0.0))
            
            # Determine time boundaries
            t_min = ann.get('start_sample', 0) // spec_obj.hop_length
            mask = ann.get('mask')
            box = ann.get('box')
            
            if mask is not None:
                t_max = t_min + mask.shape[1]
            elif box:
                t_max = box[1]
            else:
                t_max = t_min
                
            x_min = (t_min / total_frames) * duration
            x_max = (t_max / total_frames) * duration
            box_width = x_max - x_min
            
            # 1. Plot Mask
            if mask is not None:
                # If plot is log scaled, we must warp the mask rows to match
                if spec_obj.is_logscale:
                    log_scale = np.logspace(0, 1, num=height, base=10.0) - 1
                    log_scale_indices = np.clip(log_scale * (original_height - 1) / 9.0, 0, original_height - 1).astype(int)
                    mask_to_plot = mask[log_scale_indices, :]
                else:
                    mask_to_plot = mask
                    
                # Create RGBA image mapping boolean true to the class color with 40% opacity
                rgba_mask = np.zeros((*mask_to_plot.shape, 4))
                rgba_mask[mask_to_plot > 0] = (*color, 1)
                
                ax.imshow(rgba_mask, aspect='auto', origin='lower', extent=[x_min, x_max, 0, height], zorder=2)
                
            # 2. Plot Bounding Box
            if box:
                _, _, f_min, f_max = box
                
                # Map STFT frequency bins to Y-axis plot coordinates
                if spec_obj.is_logscale:
                    f_min_plot = map_freq_to_log_pixels(height, [f_min * nyquist / (original_height - 1)], nyquist)[0]
                    f_max_plot = map_freq_to_log_pixels(height, [f_max * nyquist / (original_height - 1)], nyquist)[0]
                else:
                    f_min_plot = f_min
                    f_max_plot = f_max
                    
                box_height = f_max_plot - f_min_plot
                
                rect = patches.Rectangle(
                    (x_min, f_min_plot), box_width, box_height, 
                    linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--', zorder=3
                )
                ax.add_patch(rect)
                
                # Plot Label
                if record:
                    label_text = f"{label} ({ann.get('play_idx', 0)+1})"
                    ax.text(x_min, f_max_plot + (0.02 * height), label_text, color=color, fontsize=8, 
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1), zorder=4)

    # --- Bandpass indicator lines ---
    if bandpass_hz is not None:
        hp_hz, lp_hz = bandpass_hz
        nyquist = spec_obj.sample_rate / 2
        original_height = spec_obj.n_fft // 2 + 1

        for freq_hz, label in ((hp_hz, f'HP {hp_hz} Hz'), (lp_hz, f'LP {lp_hz} Hz')):
            if freq_hz <= 0 or freq_hz >= nyquist:
                continue  # skip lines sitting on the boundary — nothing to show

            if spec_obj.is_logscale:
                y = map_freq_to_log_pixels(height, [freq_hz], nyquist)[0]
            else:
                y = freq_hz / nyquist * height

            ax.axhline(y, color='white', linestyle=':', linewidth=1.2, zorder=5)
            ax.text(
                0.01 * (spec_obj.waveform.shape[-1] / spec_obj.sample_rate),
                y + (0.01 * height),
                label, color='white', fontsize=7,
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1),
                zorder=6,
            )

    # Force the viewport to the full spectrogram dimensions
    ax.set_xlim(0, duration)
    ax.set_ylim(0, height)

    if owns_ax and show:
        plt.tight_layout()
        plt.show()

    return im

def plot_frequency_spectrum(
    spec_obj,
    ax=None,
    db_scale: bool = True,
    bandpass_hz: Optional[Tuple[int, int]] = None,
    show: bool = True,
):
    """
    Plots mean frequency spectrum (collapsed over time).
    Optionally overlays vertical bandpass cutoff lines.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        owns_ax = True
    else:
        owns_ax = False

    vals = spec_obj.values
    if spec_obj.is_complex:
        vals = vals.abs().pow(2)

    spectrum = vals[0].mean(dim=-1).cpu().numpy()
    median_spectrum = vals[0].median(dim=-1).values.cpu().numpy()

    if db_scale:
        spectrum = 10 * np.log10(spectrum + 1e-10)
        median_spectrum = 10 * np.log10(median_spectrum + 1e-10)

    nyquist = spec_obj.sample_rate / 2
    ax.plot(np.linspace(0, nyquist, len(spectrum)), spectrum, label='mean')
    ax.plot(np.linspace(0, nyquist, len(median_spectrum)), median_spectrum, label='median')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)" if db_scale else "Power")
    ax.legend()

    if bandpass_hz is not None:
        hp_hz, lp_hz = bandpass_hz

        for f, label in ((hp_hz, "HP"), (lp_hz, "LP")):
            if f and 0 < f < nyquist:
                ax.axvline(f, linestyle=":", linewidth=1.2)
                ax.text(f, ax.get_ylim()[1], f"{label} {f}Hz",
                        fontsize=8, verticalalignment='top')

    if owns_ax and show:
        plt.tight_layout()
        plt.show()

    return ax

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