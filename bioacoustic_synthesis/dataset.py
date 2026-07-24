import os
import sys
import random
import shutil
import yaml
import copy
import numpy as np
import torchaudio
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from bioacoustic_synthesis.config import load_config
from bioacoustic_synthesis.catalog import Catalog
from bioacoustic_synthesis.synthesis import SoundscapeSynthesiser
from bioacoustic_synthesis.spectrogram import Spectrogram
from bioacoustic_synthesis.spectrogram import spec_to_pil
from bioacoustic_synthesis.annotations import merge_boxes_by_class
from bioacoustic_synthesis.visualisation import plot_spectrogram
from bioacoustic_synthesis.interactive import review_sample

def generate_dataset(config_path='config.yaml', limit_per_class=None,
                     interactive=False, sample_seed=None, seed=None):
    print("=== Soundscape Dataset Generation ===")
    
    # 1. Load Configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"[Error] Failed to load config: {e}")
        sys.exit(1)

    runs = [("artificial_dataset", False)]
    if getattr(config.output, 'generate_raw_dataset', False):
        runs.append(("artificial_dataset_raw", True))

    for out_dir_name, is_raw in runs:
        print(f"\n=== Starting Run: {out_dir_name} ===")
        _generate_single_run(config, config_path, out_dir_name, is_raw,
                             limit_per_class, interactive, sample_seed, seed)

def _generate_single_run(base_config, config_path, out_dir_name, is_raw,
                         limit_per_class, interactive=False, sample_seed=None, seed=None):
    config = copy.deepcopy(base_config)
    
    if is_raw:
        config.output.include_boxes = False
        config.output.include_masks = False
        config.output.include_presence = False

    # 2. Init Catalog
    try:
        catalog = Catalog(config, limit_per_class=limit_per_class, use_raw_vocalisations=is_raw, sample_seed=sample_seed)
    except Exception as e:
        print(f"[Error] Failed to load catalog: {e}")
        return
        
    if not catalog.positives:
        print(f"[Warning] No positives found for {out_dir_name}. Skipping.")
        return

    # 3. Init Synthesiser
    bioacoustic_synthesis = SoundscapeSynthesiser(config)

    # 4. Prepare Directories
    out_dir = Path(config.paths.output) / out_dir_name
    if config.output.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    example_dir = out_dir / "example"

    out_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(config_path, out_dir / "generation_config.yaml")
    # Provenance for the run, appended rather than merged so the copied config
    # stays byte-identical to the one that was passed in.
    with open(out_dir / "generation_config.yaml", 'a') as f:
        f.write(f"\n# --- run provenance (appended by dataset.py) ---\n")
        f.write(f"seed: {seed}\n")
        f.write(f"sample_seed: {sample_seed}\n")
        f.write(f"limit_per_class: {limit_per_class}\n")
    
    for split in ['train', 'val']:
        if config.output.include_audio:
            (out_dir / f"sound_files/{split}").mkdir(parents=True, exist_ok=True)
        if config.output.include_spectrogram:
            (out_dir / f"images/{split}").mkdir(parents=True, exist_ok=True)
        if config.output.include_boxes:
            (out_dir / f"box_labels/{split}").mkdir(parents=True, exist_ok=True)
        if config.output.include_presence:
            (out_dir / f"presence/{split}").mkdir(parents=True, exist_ok=True)
        if config.output.include_simple_labels:
            (out_dir / f"labels/{split}").mkdir(parents=True, exist_ok=True)
        if config.output.include_masks:
            (out_dir / f"unetplusplus_masks/{split}/images").mkdir(parents=True, exist_ok=True)
            (out_dir / f"unetplusplus_masks/{split}/masks").mkdir(parents=True, exist_ok=True)
            (out_dir / f"unetplusplus_masks/{split}/labels").mkdir(parents=True, exist_ok=True)

    n_soundscapes = config.synthesis.n_soundscapes
    
    print(f"\n[Dataset] Beginning synthesis loop ({n_soundscapes} iterations)...")
    
    for idx in range(n_soundscapes):
        try:
            # Pick background
            bg_record = catalog.sample_background()

            n_negatives = random.randint(
                config.synthesis.negative_overlay_range[0],
                config.synthesis.negative_overlay_range[1],
            ) if catalog.negatives else 0
            negatives = [catalog.sample_negative() for _ in range(n_negatives)]

            # Pass the full positives pool; synthesis.py manages sampling and retry logic
            positives = catalog.positives
            
            # Generate the mix
            mixed_waveform, annotations = bioacoustic_synthesis.generate(bg_record, negatives, positives, catalog)
            print(f"[{idx+1}/{n_soundscapes}] Mix complete for {bg_record.path.name} with {len(negatives)} negatives and {len(annotations)} positive annotations.")

            # Determine split
            split = 'val' if idx >= int(n_soundscapes * config.output.val_ratio) else 'train'

            # 5. Save Outputs
            
            # Audio
            if config.output.include_audio:
                wav_path = out_dir / f"sound_files/{split}/{idx}.wav"
                torchaudio.save(str(wav_path), mixed_waveform.tensor.cpu(), mixed_waveform.sample_rate)

            # Spectrogram Processing
            spec = Spectrogram(mixed_waveform.tensor, mixed_waveform.sample_rate,
                               n_fft=config.spectrogram.n_fft,
                               hop_length=config.spectrogram.hop_length,
                               win_length=config.spectrogram.win_length,
                               log_base=config.spectrogram.log_base)
            spec.to_real(power=2.0)
            spec.to_logscale()

            if interactive:
                action = review_sample(mixed_waveform, spec, annotations, idx, n_soundscapes)
                if action == 'quit':
                    print("[Interactive] Quit requested — stopping pipeline.")
                    break

            if idx < 3:
                # Save example audio
                torchaudio.save(str(example_dir / f"example_{idx}.wav"), mixed_waveform.tensor.cpu(), mixed_waveform.sample_rate)
                
                # Save raw spectrogram
                fig, ax = plt.subplots(figsize=(10, 4))
                plot_spectrogram(spec, ax=ax, show=False, db_scale=True, cmap='dusk')
                fig.savefig(example_dir / f"example_{idx}_raw.png", bbox_inches='tight', dpi=150)
                plt.close(fig)
                
                # Save labeled spectrogram
                fig, ax = plt.subplots(figsize=(10, 4))
                plot_spectrogram(spec, ax=ax, show=False, db_scale=True, cmap='dusk', annotations=annotations)
                fig.savefig(example_dir / f"example_{idx}_labels.png", bbox_inches='tight', dpi=150)
                plt.close(fig)

            img = None
            if config.output.include_spectrogram or config.output.include_masks:
                img = spec_to_pil(spec.values.cpu(), resize=(640, 640), iscomplex=False, 
                                  normalise='power_to_PCEN', color_mode=config.output.color_mode)
                
                if config.output.include_spectrogram:
                    img_path = out_dir / f"images/{split}/{idx}.jpg"
                    img.save(str(img_path), quality=95)

            # Boxes, Masks, Presence, and Simple Labels Processing
            if config.output.include_boxes or config.output.include_masks or config.output.include_presence or config.output.include_simple_labels:
                original_height = spec.n_fft // 2 + 1
                height = spec.values.shape[1]
                width = spec.values.shape[2]
                
                total_samples = mixed_waveform.tensor.shape[1]
                instance_mask = np.zeros((height, width), dtype=np.uint8)
                species_mask = np.zeros((height, width), dtype=np.uint8)
                presence_array = np.zeros(1000, dtype=np.bool_) if config.output.include_presence else None
                
                def mark_presence(start_samp, end_samp):
                    if start_samp >= total_samples: return
                    end_samp = min(end_samp, total_samples)
                    if end_samp > start_samp:
                        c_start = min(999, int((start_samp / total_samples) * 1000))
                        c_end = min(999, int(((end_samp - 1) / total_samples) * 1000))
                        presence_array[c_start:c_end + 1] = True

                boxes = []
                classes = []

                log_base = config.spectrogram.log_base
                log_scale = np.logspace(0, 1, num=height, base=log_base) - 1
                log_scale_indices = np.clip(log_scale * (original_height - 1) / (log_base - 1), 0, original_height - 1).astype(int)

                instance_id = 1
                for ann in annotations:
                    record = ann.get('record')
                    if not record: continue
                    class_id = record.class_id
                    
                    t_min = ann.get('start_sample', 0) // spec.hop_length
                    ann_mask = ann.get('mask')
                    ann_box = ann.get('box')

                    if ann_mask is not None:
                        warped_mask = ann_mask[log_scale_indices, :]
                        t_max = min(t_min + warped_mask.shape[1], width)
                        w_len = t_max - t_min
                        
                        active_pixels = warped_mask[:, :w_len] > 0
                        
                        instance_mask[:, t_min:t_max][active_pixels] = instance_id
                        species_mask[:, t_min:t_max][active_pixels] = class_id + 1
                        instance_id += 1
                    
                    if presence_array is not None:
                        if ann_mask is not None:
                            active_frames = ann_mask.max(axis=0) > 0
                            active_indices = np.where(active_frames)[0]
                            for frame_idx in active_indices:
                                s_start = ann['start_sample'] + frame_idx * spec.hop_length
                                s_end = s_start + spec.hop_length
                                mark_presence(s_start, s_end)
                        else:
                            s_start = ann.get('start_sample', 0)
                            s_end = ann.get('end_sample', s_start)
                            mark_presence(s_start, s_end)

                    if ann_box:
                        _, _, f_min_linear, f_max_linear = ann_box
                        
                        rel_f_min = f_min_linear / (original_height - 1)
                        rel_f_max = f_max_linear / (original_height - 1)
                        
                        log_f_min = np.log(rel_f_min * (log_base - 1) + 1) / np.log(log_base)
                        log_f_max = np.log(rel_f_max * (log_base - 1) + 1) / np.log(log_base)
                        
                        x_min = np.clip(ann_box[0] / width, 0, 1)
                        x_max = np.clip(ann_box[1] / width, 0, 1)
                        y_min = np.clip(log_f_min, 0, 1)
                        y_max = np.clip(log_f_max, 0, 1)
                        
                        boxes.append([x_min, x_max, y_min, y_max])
                        classes.append(class_id)

                # Save Bounding Boxes (YOLO Format)
                if config.output.include_boxes:
                    yolo_boxes = []
                    if boxes:
                        merged_boxes, merged_classes = merge_boxes_by_class(boxes, classes, iou_threshold=0.1, ios_threshold=0.4, format='xxyy')
                        for box, cls in zip(merged_boxes, merged_classes):
                            x_min, x_max, y_min, y_max = box
                            
                            x_c = np.clip((x_min + x_max) / 2, 0, 1)
                            w = np.clip(x_max - x_min, 0, 1)
                            y_c = np.clip(1.0 - (y_min + y_max) / 2, 0, 1) # YOLO reverses y
                            h = np.clip(y_max - y_min, 0, 1)
                            
                            yolo_boxes.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                            
                    box_path = out_dir / f"box_labels/{split}/{idx}.txt"
                    with open(box_path, 'w') as f:
                        f.write("\n".join(yolo_boxes) + "\n")

                # Save 1D Presence Array (.npy)
                if config.output.include_presence and presence_array is not None:
                    presence_path = out_dir / f"presence/{split}/{idx}.npy"
                    np.save(str(presence_path), presence_array)

                # Save Simple Labels
                if config.output.include_simple_labels:
                    simple_classes = set()
                    for ann in annotations:
                        rec = ann.get('record')
                        if rec and rec.class_id is not None:
                            simple_classes.add(rec.class_id)
                    labels_path = out_dir / f"labels/{split}/{idx}.txt"
                    with open(labels_path, 'w') as f:
                        f.write(" ".join(str(c) for c in sorted(simple_classes)) + "\n")

                # Save U-Net++ Masks
                if config.output.include_masks and img is not None:
                    i_mask_flipped = np.flipud(instance_mask)
                    s_mask_flipped = np.flipud(species_mask)
                    
                    i_img = Image.fromarray(i_mask_flipped, 'L').resize((640, 640), Image.Resampling.NEAREST)
                    s_img = Image.fromarray(s_mask_flipped, 'L').resize((640, 640), Image.Resampling.NEAREST)
                    
                    unet_split_dir = out_dir / f"unetplusplus_masks/{split}"
                    img.save(unet_split_dir / f"images/{idx}.png", format='PNG')
                    i_img.save(unet_split_dir / f"masks/{idx}.png", format='PNG')
                    s_img.save(unet_split_dir / f"labels/{idx}.png", format='PNG')
                    
                    unet_root = out_dir / "unetplusplus_masks"
                    if not (unet_root / "generation_params.yaml").exists():
                        shutil.copy(config_path, unet_root / "generation_params.yaml")

        except Exception as e:
            print(f"[Error] Synthesis failed on iteration {idx}: {e}")
            import traceback
            traceback.print_exc()
            break

    # 6. Post-Loop Manifest Writes
    print("\n[Dataset] Writing manifests...")
    
    # species_value_map.csv
    with open(out_dir / 'species_value_map.csv', 'w') as f:
        for key, value in catalog.species_map.items():
            f.write(f"{value},{key}\n")

    # dataset.yaml for YOLO training
    names_map = {str(v): k for k, v in catalog.species_map.items()}
    yaml_data = {
        'path': '.',
        'nc': len(names_map),
        'names': names_map,
        'train': 'images/train',
        'val': 'images/val'
    }
    with open(out_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    _print_dataset_summary(catalog, n_soundscapes)


def _print_dataset_summary(catalog: 'Catalog', n_soundscapes: int):
    """Prints a terminal summary with ASCII bar chart of species proportions."""
    BAR_WIDTH = 40

    # class_counts starts at 1 per species (prior), so subtract to get actual plays
    plays = {s: max(0, count - 1) for s, count in catalog.class_counts.items()}
    total_plays = sum(plays.values())

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Soundscapes generated : {n_soundscapes}")
    print(f"  Total positive plays  : {total_plays}")
    print(f"  Species               : {len(plays)}")
    print("-" * 60)

    if total_plays > 0:
        sorted_plays = sorted(plays.items(), key=lambda x: x[1], reverse=True)
        label_w = max(len(s) for s in plays) + 2

        for species, count in sorted_plays:
            pct = count / total_plays
            filled = round(pct * BAR_WIDTH)
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            print(f"  {species:<{label_w}} {bar}  {count:>5} ({pct * 100:5.1f}%)")
    else:
        print("  (no positive annotations recorded)")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    generate_dataset()