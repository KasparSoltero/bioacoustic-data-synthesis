import torch
import torchaudio
import numpy as np
from PIL import Image
from ultralytics import YOLO
from matplotlib.colors import hsv_to_rgb
import random
import os
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
import copy

### this file needs refactoring

# manual verification workflow functions
def load_and_process_audio(file_path, images_per_file, chunk_length, overlap, resample_rate, color_mode='HSV'):
    spectrograms = load_spectrogram(file_path, max=images_per_file, chunk_length=chunk_length, overlap=overlap, resample_rate=resample_rate, unit_type='power')
    if spectrograms is None:
        return None
    
    images = []
    for spec in spectrograms:
        spec = spectrogram_transformed(spec, highpass_hz=50, lowpass_hz=16000)
        spec = spectrogram_transformed(spec, set_db=-10)
        images.append(spectrogram_transformed(spec, to_pil=True, log_scale=True, normalise='power_to_PCEN', resize=(640, 640), color_mode=color_mode))
    
    return images

# load external annoations from spreadsheet
def get_ground_truth_boxes(annotations, file_name, chunk_length):
    gt_boxes = []
    if annotations is not None:
        file_annotations = annotations[annotations['filename'] == file_name]
        for i in range(0, 60*10, chunk_length):
            specific_boxes = []
            for _, row in file_annotations.iterrows():
                if (row['start_time'] >= i and row['start_time'] <= i + chunk_length) or (row['end_time'] >= i and row['end_time'] <= i + chunk_length):
                    x_start = max(0, (row['start_time'] - i) / chunk_length)
                    x_end = min(1, (row['end_time'] - i) / chunk_length)
                    y_end, y_start = map_frequency_to_log_scale(24000, [row['freq_min'], row['freq_max']])
                    y_end = 1 - (y_end / 24000)
                    y_start = 1 - (y_start / 24000)
                    specific_boxes.append([x_start, y_start, x_end, y_end])
            gt_boxes.append(specific_boxes)
    return gt_boxes

def predict_dont_merge_boxes(model, images, conf_threshold):
    results = model.predict(images, device='mps', save=False, show=False, verbose=False, conf=conf_threshold, iou=1)
    boxes = [result.boxes.xyxyn.cpu().numpy() for result in results]
    classes = [result.boxes.cls.cpu().numpy() for result in results]
    confidences = [result.boxes.conf.cpu().numpy() for result in results]
    return boxes, classes, confidences

def predict_and_merge_boxes(model, images, conf_threshold, iou_threshold=0.1, ios_threshold=0.4):
    if isinstance(conf_threshold, list):
        set_of_merged_boxes = []
        set_of_merged_classes = []
        minimum_conf_threshold = conf_threshold[0]
        results = model.predict(images, device='mps', save=False, show=False, verbose=False, conf=minimum_conf_threshold, iou=1)
        for i, conf_threshold in enumerate(conf_threshold):
            merged_boxes = []
            merged_classes = []
            boxes = [result.boxes[result.boxes.conf >= conf_threshold].xyxyn.cpu().numpy() for result in results]
            classes = [result.boxes[result.boxes.conf >= conf_threshold].cls.cpu().numpy() for result in results]
            for image_boxes, image_classes in zip(boxes, classes):
                merged_image_boxes, merged_image_classes = merge_boxes_by_class(image_boxes, image_classes, iou_threshold=iou_threshold, ios_threshold=ios_threshold, format='xyxy')
                merged_boxes.append(merged_image_boxes)
                merged_classes.append(merged_image_classes)
            set_of_merged_boxes.append(merged_boxes)
            set_of_merged_classes.append(merged_classes)
        return set_of_merged_boxes, set_of_merged_classes
    else:
        results = model.predict(images, device='mps', save=False, show=False, verbose=False, conf=conf_threshold, iou=1)
        boxes = [result.boxes.xyxyn.cpu().numpy() for result in results]
        classes = [result.boxes.cls.cpu().numpy() for result in results]
        merged_boxes = []
        merged_classes = []
        for image_boxes, image_classes in zip(boxes, classes):
            merged_image_boxes, merged_image_classes = merge_boxes_by_class(image_boxes, image_classes, iou_threshold=iou_threshold, ios_threshold=ios_threshold, format='xyxy')
            merged_boxes.append(merged_image_boxes)
            merged_classes.append(merged_image_classes)
        return merged_boxes, merged_classes

# find predicted boxes which are outside of external annotations
def _get_verification_binaries_old(merged_boxes, gt_boxes):
    verification_binaries = []
    found_gt_binaries = []
    for image_boxes, gt_box_set in zip(merged_boxes, gt_boxes):
        image_verification_binaries = []
        for box in image_boxes:
            x1, y1, x2, y2 = box[:4]
            model_box = [x1, y1, x2, y2]
            is_inside = any(is_box_center_inside(model_box, gt_box) for gt_box in gt_box_set)
            image_verification_binaries.append(is_inside)
        verification_binaries.append(image_verification_binaries)
        image_found_gt_binaries = []
        for gt_box in gt_box_set:
            image_found_gt_binaries.append(any(is_box_center_inside(model_box, gt_box) for model_box in image_boxes))
        found_gt_binaries.append(image_found_gt_binaries)
    return verification_binaries, found_gt_binaries

def check_if_pred_boxes_are_inside(pred_boxes, gt_boxes):
    verification_binaries = []
    for (box,conf,cls) in pred_boxes:
        x1, y1, x2, y2 = box[:4]
        model_box = [x1, y1, x2, y2]
        is_inside = any(is_box_center_inside(model_box, gt_box) for gt_box in gt_boxes)
        verification_binaries.append(is_inside)
    return verification_binaries

def calculate_detection_stats(merged_boxes, merged_classes, verification_binaries, found_gt_binaries, gt_boxes, filename, num_saved):
    raw_tp_total = sum(sum(binaries) for binaries in verification_binaries)
    tp_total, fp_total, fn_total = 0, 0, 0
    fn_total = sum(sum(_) for _ in found_gt_binaries)
    fn_total = sum(len(gt_box_set) for gt_box_set in gt_boxes) - fn_total
    fp_total = sum(len(_) for _ in verification_binaries) - raw_tp_total
    human_tp_total, human_fp_total, human_fn_total = 0, 0, num_saved
    human_tp_total = sum(len(gt_box_set) for gt_box_set in gt_boxes) # all gt boxes are human tp, fp=0, all num_saved are human fn

    for image_boxes, image_classes, image_verification_binaries, gt_box_set in zip(merged_boxes, merged_classes, verification_binaries, gt_boxes):
        detected_gt = set()
        # count only the first of each gt
        for box, cls, is_verified in zip(image_boxes, image_classes, image_verification_binaries):
            if is_verified:
                x1, y1, x2, y2 = box[:4]
                model_box = [x1, y1, x2, y2]
                is_inside = False
                for i, gt_box in enumerate(gt_box_set):
                    if is_box_center_inside(model_box, gt_box):
                        is_inside = True
                        if i not in detected_gt:
                            tp_total += 1
                            detected_gt.add(i)
                        else:
                            pass # detected multiple times, do nothing for remaining times
                        break
                # if not is_inside:
                #     pass # Verified box but not inside any GT box. either new or equivalent, accounted for by num_saved (new) and raw_tp_total (inc. equivalent)

    tp_total += num_saved # Add the number of new annotations to the true positives
    
    print_detection_stats(raw_tp_total, tp_total, fp_total, fn_total, human_tp_total, human_fp_total, human_fn_total, filename)

def plot_detections(images, gt_boxes, merged_boxes, merged_classes, verification_binaries, start_idx):
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    for i, image in enumerate(images):
        ax = axs[i // 5, i % 5]
        
        ax.imshow(np.array(image))
        ax.set_xticks(np.linspace(0, 640, 6))
        ax.set_xticklabels([f'{i:.1f}' for i in np.linspace(0, 10, 6)])
        ax.set_yticks(np.linspace(0, 640, 13))
        ax.set_yticklabels([f'{i:.0f}' for i in np.linspace(24, 0, 13)])

        ax.set_title(f'{i}: {(start_idx+i)*10}s - {(start_idx+i+1)*10}s', fontsize=8)

        # Plot ground truth boxes
        for box in gt_boxes[i]:
            x1, y1, x2, y2 = [coord * 640 for coord in box]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, facecolor='black', edgecolor='black', linewidth=1)
            ax.add_patch(rect)

        # Plot merged model prediction boxes
        for j, (box, cls) in enumerate(zip(merged_boxes[i], merged_classes[i])):
            x1, y1, x2, y2 = box[:4]
            x1, y1, x2, y2 = [coord * 640 for coord in [x1, y1, x2, y2]]

            if verification_binaries[i][j]:
                color = 'red'
                linewidth = 1
            else:
                color = 'white'
                linewidth = 1

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth, linestyle='--')
            ax.add_patch(rect)
            ax.text(x1, y2+0.1, f"{i}.{j}", color='white', fontsize=8)

    # Remove any unused subplots
    for i in range(len(images), 10):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    return fig

# user inputs negatives, new annotations (model predictions outside external annotations) are saved
def process_user_input(current_boxes, current_classes, current_verification_binaries, start_idx, filename):
    audio_file = f"../data/testing/7050014/{filename}"
    audio, sample_rate = sf.read(audio_file)

    binaries_should_not_be_saved = copy.deepcopy(current_verification_binaries)
    updated_verification_binaries = copy.deepcopy(current_verification_binaries)

    while True:
        if all(all(binary for binary in binaries) for binaries in binaries_should_not_be_saved):
            print("All boxes have been verified. Continuing to the next file.")
            break
        print(f'should we save: ')
        for i, (boxes, classes, binary) in enumerate(zip(current_boxes, current_classes, binaries_should_not_be_saved)):
            for j, (box, cls, binary) in enumerate(zip(boxes, classes, binary)):
                if not binary:
                    print(f'{i}.{j}', end=', ')
        print('\r\r')

        user_input = input("'v' to verify and continue,\n 'x i.j' to exclude box,\n 'e i.j' to mark true but not save,\n 'p i.j' to play audio for box: ")
        
        if user_input.lower() == 'v':
            break
        
        elif user_input.startswith('x '):
            box_id = user_input.split()[1]
            try:
                img_idx, box_idx = map(int, box_id.split('.'))
                if 0 <= img_idx < len(current_verification_binaries) and 0 <= box_idx < len(current_verification_binaries[img_idx]):
                    binaries_should_not_be_saved[img_idx][box_idx] = True
                    print(f"Box {box_id} will not be saved as a new annotation.")
                else:
                    print(f"Warning: Invalid box number {box_id}")
            except ValueError:
                print(f"Warning: Invalid {box_id}")

        elif user_input.startswith('e '): #equivalent to a gt box; accurate but not new, update the verification binary but don't save
            box_id = user_input.split()[1]
            try:
                img_idx, box_idx = map(int, box_id.split('.'))
                if 0 <= img_idx < len(current_boxes) and 0 <= box_idx < len(current_boxes[img_idx]):
                    updated_verification_binaries[img_idx][box_idx] = True
                    binaries_should_not_be_saved[img_idx][box_idx] = True
                    print(f"Box {box_id} is equivalent to a ground truth box. not saving")
                else:
                    print(f"Warning: Invalid box number {box_id}")
            except ValueError:
                print(f"Warning: Invalid {box_id}")

        elif user_input.startswith('p '): #play audio for selected box
            box_id = user_input.split()[1]
            try:
                img_idx, box_idx = map(int, box_id.split('.'))
                if 0 <= img_idx < len(current_boxes) and 0 <= box_idx < len(current_boxes[img_idx]):
                    box = current_boxes[img_idx][box_idx]
                    start_time = (start_idx + img_idx) * 10 + (box[0] * 10)
                    start_time = max(0, start_time-1)  # Start 1 second earlier
                    end_time = (start_idx + img_idx) * 10 + (box[2] * 10)
                    end_time = min(len(audio) / sample_rate, end_time+1)  # End 1 second later
                    # Extract and play the audio segment
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    audio_segment = audio[start_sample:end_sample]
                    # Convert audio to int16 if it's float
                    if audio_segment.dtype == np.float32 or audio_segment.dtype == np.float64:
                        audio_segment = (audio_segment * 32767).astype(np.int16)

                    # Convert to AudioSegment and play
                    audio_segment_pydub = AudioSegment(
                        audio_segment.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=audio_segment.dtype.itemsize,
                        channels=1 if len(audio_segment.shape) == 1 else audio_segment.shape[1]
                    )
                    play(audio_segment_pydub)
                    
                    print(f"Played audio for box {user_input}")
                else:
                    print(f"Warning: Invalid box number {user_input}")
            except ValueError as e:
                print(f"Warning: Invalid input format for box number {user_input}, {e}")

        elif user_input.lower() == 'q':
            print("skipping this file")
            return None, None

    # Save new annotations for all boxes that are still marked as outside ground truth
    num_saved = 0
    for i, (boxes, classes, binaries) in enumerate(zip(current_boxes, current_classes, binaries_should_not_be_saved)):
        for j, (box, cls, binary) in enumerate(zip(boxes, classes, binaries)):
            if not binary:  # If the box is still marked as should-be-saved (False)
                # Save as new annotation
                print(f'saving box: {i}.{j}, {box}')
                start_time = (start_idx + i) * 10 + (box[0] * 10)
                end_time = (start_idx + i) * 10 + (box[2] * 10)
                start_freq, end_freq = (1 - box[3]) * 24000, (1 - box[1]) * 24000
                start_freq, end_freq = map_frequency_to_linear_scale(24000, [start_freq, end_freq])
                save_new_annotation(filename, start_time, end_time, start_freq, end_freq, cls)
                num_saved += 1
                
                # Update verification binary to True (verified)
                updated_verification_binaries[i][j] = True

    return updated_verification_binaries, num_saved

def save_new_annotation(filename, start_time, end_time, start_freq, end_freq, cls):
    with open('new_manual_annotations.txt', 'a') as f:
        f.write(f"{filename}, {start_time:.3f}, {end_time:.3f}, {start_freq:.0f}, {end_freq:.0f}, {cls}\n")

def calculate_f1_score(tp, fp, fn):
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    return precision, recall, f1_score

def print_detection_stats(raw_tp, tp, fp, fn, human_tp, human_fp, human_fn, filename):
    precision, recall, f1_score = calculate_f1_score(tp, fp, fn)
    human_precision, human_recall, human_f1_score = calculate_f1_score(human_tp, human_fp, human_fn)

    raw_precision = raw_tp / (raw_tp + fp) if raw_tp + fp > 0 else 0
    raw_recall = raw_tp / (raw_tp + fn) if raw_tp + fn > 0 else 0
    raw_f1_score = 2 * (raw_precision * raw_recall) / (raw_precision + raw_recall) if raw_precision + raw_recall > 0 else 0
    total_detections = tp + fn

    print(f"\nFile {filename}: {total_detections} detections, {raw_recall}% recall\n ({tp} {fp} {fn})\n ({raw_precision:.2f} {raw_recall:.2f} {raw_f1_score:.2f}) raw\n ({precision:.2f} {recall:.2f} {f1_score:.2f}) simple\n ({human_precision:.2f} {human_recall:.2f} {human_f1_score:.2f}) human")

    save_stats_file = 'detection_stats.txt'
    with open(save_stats_file, 'a') as f:
        f.write(f"{filename}, {total_detections}, {raw_tp}, {tp}, {fp}, {fn}, {human_tp}, {human_fp}, {human_fn}, {precision:.3f}, {recall:.3f}, {f1_score:.3f}, {human_precision:.3f}, {human_recall:.3f}, {human_f1_score:.3f}, {raw_precision:.3f}, {raw_recall:.3f}, {raw_f1_score:.3f}\n")
    
def print_these_detection_stats(file_idx, total_files, start_idx, end_idx, total_images, current_boxes, current_verification_binaries, current_gt_binaries, current_gt_boxes):
    total_detections = sum(len(box) for box in current_boxes)
    total_verified = sum(sum(binaries) for binaries in current_verification_binaries)
    print(f'')

    print(f"\nFile {file_idx + 1}/{total_files}: Images {start_idx + 1}-{end_idx} of {total_images}. ")
    
    # Print detection counts for each image
    if total_detections > 0:
        print(f"{total_detections} detections {100*total_verified/total_detections:.2f}% in {sum(sum(_) for _ in current_gt_binaries)} of {sum(len(_) for _ in current_gt_boxes)} gt boxes", end='')
    else:
        print("No detections")
    for i, boxes in enumerate(current_boxes): # 5 by 2 detection number grid
        if i % 5 == 0:
            print()  # Start a new line after every 5 numbers
        print(f"{len(boxes):2d}", end=" ")
    print("\n" + "="*50 + "\n")

# box section checks. ALL XYXY
def is_box_inside(box1, box2):
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2
    return (X1 <= x1 <= X2 and X1 <= x2 <= X2 and
            Y1 <= y1 <= Y2 and Y1 <= y2 <= Y2)

def is_box_center_inside(box1, box2):
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    return X1 <= x_center <= X2 and Y1 <= y_center <= Y2

def convert_box_to_time_freq(box, image_duration=10, max_freq=24000):
    x1, y1, x2, y2 = box
    start_time = x1 * image_duration
    end_time = x2 * image_duration
    # Note: y-axis is inverted in the image
    start_freq = (1 - y2) * max_freq
    end_freq = (1 - y1) * max_freq
    return start_time, end_time, start_freq, end_freq

# merge boxes. both xxyy (generate dataset) and xyxy (process audio)
def calculate_iou_ios(box1, box2, format):
            # Coordinates of the intersection rectangle
            if format=='xxyy':
                x_left = max(box1[0], box2[0])
                y_top = max(box1[2], box2[2])
                x_right = min(box1[1], box2[1])
                y_bottom = min(box1[3], box2[3])
                box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
                box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
                intersection_area = (x_right-x_left) * (y_bottom-y_top)
                # No intersection
                if x_right < x_left or y_bottom < y_top:
                    return 0.0, 0
            elif format=='xyxy':
                x_left = max(box1[0], box2[0])
                y_bottom = max(box1[1], box2[1])
                x_right = min(box1[2], box2[2])
                y_top = min(box1[3], box2[3])
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                intersection_area = (x_right-x_left) * (y_top-y_bottom)
                # No intersection
                if x_right < x_left or y_top < y_bottom or box1_area == 0 or box2_area == 0:
                    return 0.0, 0

            ios = intersection_area / min(box1_area, box2_area)
            # Calculate IoU
            iou = intersection_area / float(box1_area + box2_area - intersection_area)
            return iou, ios

def combine_boxes(box1, box2, format='xxyy'):
    if format=='xxyy':
        x_min = min(box1[0], box2[0])
        x_max = max(box1[1], box2[1])
        y_min = min(box1[2], box2[2])
        y_max = max(box1[3], box2[3])
        return [x_min, x_max, y_min, y_max]
    elif format=='xyxy':
        x_min = min(box1[0], box2[0])
        x_max = max(box1[2], box2[2])
        y_min = min(box1[1], box2[1])
        y_max = max(box1[3], box2[3])
        return [x_min, y_min, x_max, y_max]

def find(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i])  # Path compression
    return parent[i]

def union(parent, rank, i, j):
    root_i = find(parent, i)
    root_j = find(parent, j)
    
    if root_i != root_j:
        if rank[root_i] > rank[root_j]:
            parent[root_j] = root_i
        elif rank[root_i] < rank[root_j]:
            parent[root_i] = root_j
        else:
            parent[root_j] = root_i
            rank[root_i] += 1

def merge_boxes_by_class(boxes, classes, iou_threshold=0.5, ios_threshold=0.5, format='xxyy'):
    parent = list(range(len(boxes)))
    rank = [0] * len(boxes)
    
    for i, (box, species_class) in enumerate(zip(boxes, classes)):
        for k in range(len(boxes) - 1, -1, -1):
            if k == i:
                continue
            if species_class != classes[k]:
                continue
            other_box = boxes[k]
            iou, ios = calculate_iou_ios(box, other_box, format)
            if iou > iou_threshold or ios > ios_threshold:
                union(parent, rank, i, k)
    
    merged_boxes = {}
    for i in range(len(boxes)):
        root = find(parent, i)
        if root not in merged_boxes:
            merged_boxes[root] = boxes[i]
        else:
            merged_boxes[root] = combine_boxes(merged_boxes[root], boxes[i], format)
    
    # Propagate the merge to ensure indirect overlaps are included
    updated = True
    while updated:
        updated = False
        temp_merged_boxes = merged_boxes.copy()
        for root1 in list(temp_merged_boxes.keys()):
            for root2 in list(temp_merged_boxes.keys()):
                if root1 == root2:
                    continue
                if classes[root1] != classes[root2]:  # Ensure same class
                    continue
                iou, ios = calculate_iou_ios(temp_merged_boxes[root1], temp_merged_boxes[root2], format)
                if iou > iou_threshold or ios > ios_threshold:
                    temp_merged_boxes[root1] = combine_boxes(temp_merged_boxes[root1], temp_merged_boxes[root2], format)
                    del temp_merged_boxes[root2]
                    updated = True
                    break
            if updated:
                break
        merged_boxes = temp_merged_boxes

    # Extract the final merged boxes and their corresponding classes
    final_boxes = list(merged_boxes.values())
    final_classes = []
    for root in merged_boxes:
        final_classes.append(classes[root])
    
    return final_boxes, final_classes

# unused
def load_spectrogram_chunks(path, chunk_length=10, overlap=0.5, resample_rate=48000):
    waveform, sample_rate = torchaudio.load(path)
    resample = torchaudio.transforms.Resample(sample_rate, resample_rate)
    waveform = resample(waveform)

    # Calculate the number of samples per chunk
    samples_per_chunk = int(chunk_length * resample_rate)

    # Calculate the number of samples to overlap
    overlap_samples = int(samples_per_chunk * overlap)

    # Split the waveform into n 10-second chunks with 50% overlap
    length = waveform.shape[1]
    n = int(2*length / (samples_per_chunk)) - 1 # only works for 0.5 overlap
    if n>20: n=20
    chunks = []
    start = 0
    end = samples_per_chunk
    for _ in range(n):
        chunk = waveform[:, start:end]
        chunks.append(chunk)
        start += samples_per_chunk - overlap_samples
        end += samples_per_chunk - overlap_samples

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=2048, 
        win_length=2048, 
        hop_length=512, 
        power=2.0
    )
    return [spec_transform(chunk) for chunk in chunks]

# masks
def generate_masks(overlay_waveform, image_id, category_id, last_box, threshold_db=10, log_scale=True, freq_bounds=None, debug=False):
    """
    Generate masks from positive overlay spectrogram.
    Args:
        positive_spec: Spectrogram of positive overlay (freq x time)
    """
    overlay_spec = transform_waveform(overlay_waveform, to_spec='power')
    if freq_bounds is not None:
        # zero values outside bounds - careful here, freq bounds are in bins which depends on similarity of earlier power spec transform
        freq_min, freq_max = freq_bounds
        overlay_spec[:, :freq_min] = 0
        overlay_spec[:, freq_max:] = 0

    if log_scale:
        overlay_spec = log_scale_spectrogram(overlay_spec)

    # Generate binary mask
    spec_db = 10 * torch.log10(overlay_spec + 1e-10)
    mask = (spec_db > threshold_db).numpy().astype(np.uint8)
    # remove batch dimension
    mask = mask[0]
    if debug:
        print(f'Mask shape: {mask.shape}')
        # plot display mask, vocalisation, mask @ threshold 0
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(mask, aspect='auto', origin='lower')
        axs[0].set_title('Mask')
        axs[1].imshow(spec_db[0], aspect='auto', origin='lower')
        axs[1].set_title('Spectrogram (dB)')
        mask0 = (spec_db > -10).numpy().astype(np.uint8)[0]
        axs[2].imshow(mask0, aspect='auto', origin='lower')
        axs[2].set_title('Mask @ -10 dB')
        plt.show()

    # Create COCO annotation
    rle = rle_encode(mask)

    # convert bbox to COCO
    coco_bbox = [
        last_box[0],  # x
        last_box[2],  # y
        last_box[1] - last_box[0],  # width
        last_box[3] - last_box[2]   # height
    ]
    
    annotation = {
        'segmentation': [{
            'counts': rle,
            'size': list(mask.shape)  # [freq, time]
        }],
        'area': float(mask.sum()),
        'iscrowd': 0,
        'image_id': image_id,
        'bbox': coco_bbox,
        'category_id': category_id
    }
    
    return annotation

def rle_encode(mask):
    """
    Convert binary mask to standard RLE encoding.
    Returns: List of integers representing counts of 0s and 1s alternately,
             starting with count of 0s.
    """
    # Ensure mask is flattened
    pixels = mask.flatten()
    
    # Initialize results
    counts = []
    count = 0
    last_value = 0  # Start counting zeros
    
    # Count runs
    for pixel in pixels:
        if pixel == last_value:
            count += 1
        else:
            counts.append(count)
            count = 1
            last_value = pixel
    
    # Append final count
    counts.append(count)
    
    # If we ended with 1s, add a 0 count
    if last_value == 1:
        counts.append(0)
        
    return counts

def rle_decode(counts, shape):
    """
    Decode standard RLE encoding back to binary mask.
    counts: List of integers representing alternate counts of 0s and 1s
    shape: Target shape of the mask (height, width)
    """
    # Initialize flat mask
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    
    # Current position in the mask
    idx = 0
    
    # Fill runs
    for i, count in enumerate(counts):
        value = (i % 2) == 1  # odd indices are 1s, even indices are 0s
        if count > 0:
            mask[idx:idx + count] = value
            idx += count
    
    return mask.reshape(shape)


def spec_to_image(spectrogram, image_normalise=0.2):
    # make ipeg image
    spec = np.squeeze(spectrogram.numpy())
    # spec = (spec - np.min(spec)) / (np.ptp(spec)) #normalise to 0-1 # the max is typically an outlier so this normalisation is destructive
    spec = spec * image_normalise / spec.mean() # normalise to 0.2 mean
    spec = np.flipud(spec) #vertical flipping for image cos

    image_width, image_height = spec.shape[1], spec.shape[0]
    image = Image.fromarray(np.uint8(spec * 255), 'L')
    image = image.resize((640, 640), Image.Resampling.LANCZOS) #resize for yolo
    return image

def get_detections(paths, model_no):
    model = YOLO(f'spectral_detector/runs/detect/train{model_no}/weights/best.pt')
    all_boxes = []
    for path in paths:
        spectrograms = load_spectrogram_chunks(path)
        images = [spec_to_image(spec) for spec in spectrograms]
        results = model.predict(images, save=False, show=False, device='mps')
        boxes = []
        for i, result in enumerate(results):
            for box_set in result.boxes: 
                for box in box_set:
                    x1, y1, x2, y2 = np.squeeze(box.xyxyn.to('cpu'))
                    x1 = ((5*i)+(float(x1)*10)) / 55
                    x2 = ((5*i)+(float(x2)*10)) / 55
                    y1 = 1-y1
                    y2 = 1-y2
                    boxes.append([x1, y1, x2, y2])
        all_boxes.append(boxes)
    return all_boxes

def pcen(spec, s=0.025, alpha=0.01, delta=0, r=0.05, eps=1e-6):
    """
    Apply Per-Channel Energy Normalization (PCEN) to a spectrogram.
    
    Parameters:
    - spec (Tensor): Input spectrogram.
    - s (float): Time constant.
    - alpha (float): Exponent for the smooth function.
    - delta (float): Bias term.
    - r (float): Root compression parameter.
    - eps (float): Small value to prevent division by zero.
    
    Returns:
    - Tensor: PCEN-processed spectrogram.
    """
    try:
        if spec.ndim == 4:
            # Original shape for 4D tensor [batch, channels, frequencies, time]
            orig_shape = spec.shape
            
            # Flatten batch and channel dimensions
            flattened_spec = spec.view(orig_shape[0] * orig_shape[1], 1, orig_shape[2], orig_shape[3])
            
            # Apply avg_pool1d along the time dimension (assuming last dimension is time)
            M = torch.nn.functional.avg_pool1d(flattened_spec.flatten(start_dim=2), 
                                            kernel_size=int(s * spec.size(-1)), 
                                            stride=1, 
                                            padding=int((s * spec.size(-1) - 1) // 2)).view(orig_shape)
            
            pcen_spec = (spec / (M + eps).pow(alpha) + delta).pow(r) - delta**r
        
        elif spec.ndim == 3:
            # Original shape for 3D tensor [batch, frequencies, time]
            orig_shape = spec.shape
            
            # Unsqueeze to add a channel dimension
            spec = spec.unsqueeze(1)
            
            # Apply avg_pool1d along the time dimension
            M = torch.nn.functional.avg_pool1d(spec.flatten(start_dim=2), 
                                            kernel_size=int(s * spec.size(-1)), 
                                            stride=1, 
                                            padding=int((s * spec.size(-1) - 1) // 2)).view(orig_shape[0], 1, orig_shape[1], orig_shape[2])
            
            # Remove the added channel dimension
            pcen_spec = (spec / (M + eps).pow(alpha) + delta).pow(r) - delta**r
            pcen_spec = pcen_spec.squeeze(1)  # Removing the channel dimension
            
        else:
            raise ValueError(f"Input tensor must be either 3D or 4D, but got {spec.ndim}D tensor.")
    except Exception as e:
        print(e)
        print('pcen failed')
        return spec
    
    return pcen_spec

def log_scale_spectrogram(spec):
    #logarithmic scaling on the y-axis
    # Get original dimensions
    _, original_height, original_width = spec.shape
    log_scale = torch.logspace(0, 1, steps=original_height, base=10.0) - 1
    log_scale_indices = torch.clamp(log_scale * (original_height - 1) / (10 - 1), 0, original_height - 1).long()
    log_spec = spec[:, log_scale_indices, :]# Resample spectrogram on new y-axis

    return log_spec

def map_frequency_to_log_scale(original_height, freq_indices):
    # Convert frequency indices to log scale
    log_freq_indices = []
    for freq_index in freq_indices:
        # Find the relative position in the original linear scale
        relative_position = freq_index / (original_height - 1 if original_height > 1 else 1)
        
        # Map to the log scale
        log_position = torch.log10(torch.tensor(relative_position * (10 - 1) + 1))
        log_index = int(torch.round(log_position * (original_height - 1) / torch.log10(torch.tensor(10.0))))
        log_freq_indices.append(log_index)
    
    return log_freq_indices

def map_frequency_to_linear_scale(original_height, freq_indices):
    # Convert frequency indices from log scale to linear scale
    linear_freq_indices = []
    for freq_index in freq_indices:
        # Find the relative position in the log scale
        relative_position = freq_index / (original_height - 1 if original_height > 1 else 1)
        
        # Map from log scale to linear scale
        linear_position = (10 ** (relative_position * torch.log10(torch.tensor(10.0))) - 1) / 9
        linear_index = int(torch.round(linear_position * (original_height - 1)))
        linear_freq_indices.append(linear_index)
    
    return linear_freq_indices

def spec_to_pil(spec, resize=None, iscomplex=False, normalise='power_to_PCEN', color_mode='HSV'):
    if iscomplex:
        spec = torch.abs(spec)

    if normalise:
        if normalise == 'power_to_dB':
            spec = 10 * torch.log10(spec + 1e-6)
        elif normalise == 'dB_to_power':
            spec = 10 ** (spec / 10)
        elif normalise == 'power_to_PCEN':
            spec = pcen(spec)
        elif normalise == 'complex_to_PCEN':
            # square complex energy for magnitude power
            spec = torch.square(spec)
            spec = pcen(spec)

    spec = np.squeeze(spec.numpy())
    spec = np.flipud(spec)  # vertical flipping for image cos
    spec = (spec - spec.min()) / (spec.max() - spec.min())  # scale to 0 - 1

    if color_mode == 'HSV':
        value = spec
        saturation = 4 * value * (1 - value)
        hue = np.linspace(0,1,spec.shape[0])[:, np.newaxis] # linearly spaced hues over frequency range
        hue = np.tile(hue, (1, spec.shape[1]))
        hsv_spec = np.stack([hue, saturation, value], axis=-1)
        rgb_spec = hsv_to_rgb(hsv_spec)
        rgb_spec = np.clip(rgb_spec, 0, 1) # should already be here but ensure nonetheless
        spec = Image.fromarray(np.uint8(rgb_spec * 255), 'RGB')
    else:
        spec = Image.fromarray(np.uint8(spec * 255), 'L')

    if resize:
        spec = spec.resize(resize, Image.Resampling.LANCZOS)
    
    return spec

def spec_to_audio(spec, energy_type='power', save_to=None, normalise_rms=0.05, sample_rate=48000, specconfig={
        'n_fft': 2048,
        'win_length': 2048,
        'hop_length': 1024,
    }):
    # convert back to waveform and save to wav for viewing testing
    waveform_transform = torchaudio.transforms.GriffinLim(
        n_fft=specconfig['n_fft'],
        win_length=specconfig['win_length'],
        hop_length=specconfig['hop_length'],
        power=2.0
    )
    # if energy_type=='dB':
    #     normalise = 'dB_to_power'
    # elif energy_type=='power':
    #     normalise = None
    # spec_audio = spectrogram_transformed(spec, normalise=normalise, to_torch=True)
    # if energy_type=='complex':
    waveform = waveform_transform(spec)
    rms = torch.sqrt(torch.mean(torch.square(waveform)))
    waveform = waveform*normalise_rms/rms
    if save_to:
        # check if save_to dir exists
        if not os.path.exists(os.path.dirname(save_to)):
            # save to includes the filename
            os.makedirs(os.path.dirname(save_to))
        torchaudio.save(f"{save_to}.wav", waveform, sample_rate=sample_rate)

def band_pass_spec(spec, sample_rate=48000, lowpass_hz=None, highpass_hz=None):
    if lowpass_hz:
        lowpass = int(lowpass_hz*spec.shape[1]/(sample_rate/2))
        spec[:, lowpass:, :] = 0
    if highpass_hz:
        highpass = int(highpass_hz*spec.shape[1]/(sample_rate/2))
        spec[:, :highpass, :] = 0
    return spec

def clamp_intensity_spec(spec, clamp_intensity):
    if isinstance(clamp_intensity, (int, float)):
        spec = spec.clamp(min=clamp_intensity)
    elif isinstance(clamp_intensity, (tuple, list)):
        spec = spec.clamp(min=clamp_intensity[0], max=clamp_intensity[1])
    return spec

def scale_intensity_spec(spec, scale_intensity):
    if isinstance(scale_intensity, (int, float)):
        spec = spec * scale_intensity
    elif isinstance(scale_intensity, (tuple, list)):
        minimum = scale_intensity[0]
        maximum = scale_intensity[1]
        spec = (spec - spec.min()) / (spec.max() - spec.min()) * (maximum - minimum) + minimum
    return spec

def set_dB(spec, set_db, unit_type='energy'):
    if unit_type == 'energy':
        power = torch.mean(torch.square(spec))
        normalise_power = torch.sqrt(10 ** (set_dB / 10) / power)
    elif unit_type == 'power':
        power = torch.mean(spec)
        normalise_power = 10 ** (set_db / 10) / power
    spec = spec * normalise_power
    return spec

def complex_spectrogram_transformed(
        spec,
        lowpass=None,
        highpass=None,
        scale_intensity=None,
        clamp_intensity=None,
        set_snr_db=None,
        scale_mean=None,
        set_rms=None,
        normalise=None,
        to_torch=False,
        to_numpy=False,
        to_pil=False,
        resize=None
    ):
    # scaling and clamping is done BEFORE normalisation

    real_part = spec.real
    imaginary_part = spec.imag

    if highpass:
        hp = int(highpass*spec.shape[1]/24000)
        real_part[:, :hp, :] = 0
        imaginary_part[:, :hp, :] = 0
    if lowpass:
        lp = int(lowpass*spec.shape[1]/24000)
        real_part[:, lp:, :] = 0
        imaginary_part[:, lp:, :] = 0

    if isinstance(scale_intensity, int) or isinstance(scale_intensity, float) or (isinstance(scale_intensity, torch.Tensor) and scale_intensity.shape==[]): 
        real_part = real_part * scale_intensity
        imaginary_part = imaginary_part * scale_intensity
    elif isinstance(scale_intensity, tuple) or isinstance(scale_intensity, list):
        minimum = scale_intensity[0]
        maximum = scale_intensity[1]
        real_part = (real_part - real_part.min()) / (real_part.max() - real_part.min()) * (maximum - minimum) + minimum
        imaginary_part = (imaginary_part - imaginary_part.min()) / (imaginary_part.max() - imaginary_part.min()) * (maximum - minimum) + minimum
    if isinstance(clamp_intensity, int) or isinstance(clamp_intensity, float):
        real_part = real_part.clamp(min=clamp_intensity)
        imaginary_part = imaginary_part.clamp(min=clamp_intensity)
    elif isinstance(clamp_intensity, tuple) or isinstance(clamp_intensity, list):
        real_part = real_part.clamp(min=clamp_intensity[0], max=clamp_intensity[1])
        imaginary_part = imaginary_part.clamp(min=clamp_intensity[0], max=clamp_intensity[1])
    if set_snr_db:
        power = torch.mean(torch.sqrt(real_part**2 + imaginary_part**2))
        # set_snr to db
        set_snr = 10 ** (set_snr_db / 10)
        normalise_power = set_snr / power
        real_part = real_part * normalise_power
        imaginary_part = imaginary_part * normalise_power
    
    if scale_mean:
        # normalise mean value for the non-zero values
        non_zero_real = real_part[real_part != 0]
        mean_power = non_zero_real.mean()
        normalise_power = scale_mean / mean_power
        real_part = real_part * normalise_power
        imaginary_part = imaginary_part * normalise_power
    if set_rms:
        component_magnitude = torch.sqrt(real_part**2 + imaginary_part**2)
        rms = torch.sqrt(torch.mean(component_magnitude**2))
        normalise_rms_factor = set_rms / rms
        real_part = real_part * normalise_rms_factor
        imaginary_part = imaginary_part * normalise_rms_factor

    # if normalise:
    #     if normalise=='power_to_dB':
    #         real_part = 10 * torch.log10(real_part + 1e-6)
    #         imaginary_part = 10 * torch.log10(imaginary_part + 1e-6)
    #     elif normalise=='dB_to_power':
    #         real_part = 10 ** (real_part / 10)
    #         imaginary_part = 10 ** (imaginary_part / 10)
    #     elif normalise=='PCEN':
    #         real_part = pcen(real_part)
    #         imaginary_part = pcen(imaginary_part)

    if to_pil:
        return spec_to_pil(real_part + 1j * imaginary_part, resize=resize, iscomplex=True, normalise='complex_to_PCEN', color_mode='HSV')

    if to_numpy:
        return np.stack([real_part.numpy(), imaginary_part.numpy()], axis=-1)
    return real_part + 1j * imaginary_part

def spectrogram_transformed(
        spec,
        lowpass_hz=None, 
        highpass_hz=None, 
        scale_intensity=None, 
        clamp_intensity=None, 
        set_db=None,
        scale_mean=None, 
        set_rms=None,
        to_torch=False, 
        to_numpy=False, 
        to_pil=False, 
        normalise='power_to_PCEN', #only if to pil
        color_mode='HSV', #only if to pil
        log_scale=False,
        resize=None
    ):
    # [spec.shape] = [batch, freq_bins, time_bins]
    # scaling and clamping is done BEFORE normalisation
    is_numpy = isinstance(spec, np.ndarray)
    if is_numpy:
        to_numpy=True
        spec = torch.from_numpy(spec)

    if len(spec.shape) == 2:
        spec = spec.unsqueeze(0)

    if spec.is_complex():
        print('complex spectrogram')
        # spec = complex_spectrogram_transformed(
        #     spec,lowpass=lowpass,highpass=highpass,scale_intensity=scale_intensity,set_snr_db=set_snr_db,clamp_intensity=clamp_intensity,scale_mean=scale_mean,set_rms=set_rms,normalise=normalise,to_torch=to_torch,to_numpy=to_numpy,to_pil=to_pil,resize=resize
        # )
    else:
        if highpass_hz:
            spec = band_pass_spec(spec, highpass_hz=highpass_hz)
        if lowpass_hz:
            spec = band_pass_spec(spec, lowpass_hz=lowpass_hz)
        if scale_intensity:
            spec = scale_intensity_spec(spec, scale_intensity)
        if clamp_intensity:
            spec = clamp_intensity_spec(spec, clamp_intensity)
        if set_db:
            spec = set_dB(spec, set_db, unit_type='power')
        if log_scale:
            spec = log_scale_spectrogram(spec)

        if scale_mean:
            print('tried tp scale mean')
            # normalise mean value for the non-zero values
            # non_zero_spec = spec[spec != 0]
            # mean_power = non_zero_spec.mean()
            # normalise_power = scale_mean / mean_power
            # spec = spec * normalise_power
        if set_rms:
            print('tried to set rms')
            # assuming unit_type=power
            # power = torch.mean(spec)
            # normalise_power = set_rms / power
            # spec = spec * normalise_power

        if to_pil:
            # spec = np.squeeze(spec.numpy())
            # spec = np.flipud(spec) # vertical flipping for image cos
            # spec = (spec - spec.min()) / (spec.max() - spec.min()) # scale to 0 - 1
            # if to_pil=='HSV':
            #     value = spec
            #     saturation = 4 * value * (1 - value)
            #     hue = np.linspace(0,1,spec.shape[0])[:, np.newaxis] # linearly spaced hues over frequency range
            #     hue = np.tile(hue, (1, spec.shape[1]))
            #     hsv_spec = np.stack([hue, saturation, value], axis=-1)
            #     rgb_spec = hsv_to_rgb(hsv_spec)
            #     spec = Image.fromarray(np.uint8(rgb_spec * 255), 'RGB')
            # else: # greyscale
            #     spec = Image.fromarray(np.uint8(spec * 255), 'L')
            # if resize:
            #     spec = spec.resize(resize, Image.Resampling.LANCZOS)
            # return spec
            return spec_to_pil(spec, resize=resize, iscomplex=False, normalise=normalise,color_mode=color_mode)
    
    if to_numpy:
        spec = np.squeeze(spec.numpy())
    return spec

def load_spectrogram(
        paths, 
        unit_type='complex', 
        rms_normalised=1, 
        random_crop=None, 
        chunk_length=None, 
        max=20,
        overlap=0.5, 
        resample_rate=48000
    ):
    
    if unit_type == 'complex':
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            power=None,  # Produces complex-valued spectrogram without reducing to power or magnitude
    )
    elif unit_type=='power':
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=2048, 
            win_length=2048, 
            hop_length=512, 
            power=2.0,
            window_fn=torch.hamming_window
        )
    elif unit_type=='dB':
        spec_transform = lambda x: 10 * torch.log10(torch.abs(torchaudio.transforms.Spectrogram(
            n_fft=2048, 
            win_length=2048, 
            hop_length=512, 
            power=2.0,
            window_fn=torch.hamming_window
        )(x)) + 1e-6)

    specs = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        try:
            waveform, sample_rate = torchaudio.load(path)
        except:
            print(f'couldn"t load {path}')
            return None
        # print(f'loaded {os.path.basename(path)}, sample rate {sample_rate}')
        resample = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=torch.float32).to(waveform.device)
        waveform = resample(waveform)
        if waveform.shape[0]>1:
            waveform = torch.mean(waveform, dim=0, keepdim=True) # mono

        #randomly crop to random_crop seconds
        if random_crop: 
            cropped_samples = random_crop * resample_rate
            if waveform.shape[1] > cropped_samples:
                start = random.randint(0, waveform.shape[1] - cropped_samples)
                waveform = waveform[:, start:start+cropped_samples]
            else:
                print(f"Error: {path} is shorter than random crop length {random_crop}")
                return None
        
        # rms normalise
        if rms_normalised:
            rms = torch.sqrt(torch.mean(torch.square(waveform)))
            waveform = waveform*rms_normalised/rms

        # separate into chunks of chunk_length seconds
        if chunk_length and (waveform.shape[1] > (chunk_length * resample_rate)):
            samples_per_chunk = int(chunk_length * resample_rate)
            overlap_samples = int(samples_per_chunk * overlap)
            samples_overlap_difference = samples_per_chunk - overlap_samples
            for i in range(0, waveform.shape[1], samples_overlap_difference):
                chunk = waveform[:, i:i+samples_per_chunk]
                spec = spec_transform(chunk)
                specs.append(spec)
                if max and len(specs)>=max:
                    break

        else:
            spec = spec_transform(waveform)
            specs.append(spec)
    
    if len(specs)==1:
        return specs[0]
    return specs

def crop_overlay_waveform(
        bg_shape,
        segment,
        minimum_samples_present=48000,
    ):
    # determine the segments start position, and crop it if it extends past the background
    if segment.shape[1] > bg_shape:
        minimum_start = bg_shape - segment.shape[1]
        maximum_start = 0
    else:
        minimum_start = min(0, minimum_samples_present-segment.shape[1])
        maximum_start = max(bg_shape-segment.shape[1], bg_shape-minimum_samples_present)
    start = random.randint(minimum_start, maximum_start)
    cropped_segment = segment[:, max(0,-start) : min(bg_shape-start, segment.shape[1])]
    return cropped_segment, start

def add_noise_to_waveform(waveform, noise_power, noise_type):
    if noise_type=='white':
        noise = torch.randn_like(waveform) * torch.sqrt(torch.tensor(noise_power))
    elif noise_type=='brown':
        noise = torch.cumsum(torch.randn_like(waveform), dim=-1)
        noise = noise - noise.mean()
        noise = noise / noise.std()
        noise = noise * torch.sqrt(torch.tensor(noise_power))
    elif noise_type=='pink':
        white_noise = torch.randn_like(waveform)
        fft = torch.fft.rfft(white_noise, dim=-1)
        frequencies = torch.fft.rfftfreq(waveform.shape[-1], d=1.0)
        
        # Avoid dividing by zero frequency
        fft[:, 1:] /= torch.sqrt(frequencies[1:])
        
        pink_noise = torch.fft.irfft(fft, n=waveform.shape[-1], dim=-1)
        pink_noise = pink_noise / pink_noise.std()
        noise = pink_noise * torch.sqrt(torch.tensor(noise_power))
    return waveform + noise

def transform_waveform(waveform, 
        resample=[48000,48000], 
        specconfig={
            'n_fft':2048,
            'win_length':2048, 
            'hop_length':512
        },
        random_crop_seconds=None, 
        crop_seconds=None,
        rms_normalised=None, 
        set_db=None,
        to_spec=None, 
        add_white_noise=None, 
        add_pink_noise=None, 
        add_brown_noise=None
    ):
    if waveform.shape[0]>1:
        waveform = torch.mean(waveform, dim=0, keepdim=True) # mono
    
    if  not (resample[0]==resample[1]):
        resample_transform = torchaudio.transforms.Resample(resample[0], resample[1], dtype=torch.float32).to(waveform.device)
        waveform = resample_transform(waveform)
    
    if random_crop_seconds: 
        cropped_samples = random_crop_seconds * resample[1]
        if waveform.shape[1] > cropped_samples:
            start = random.randint(0, waveform.shape[1] - cropped_samples)
            waveform = waveform[:, start:start+cropped_samples]
        else:
            print(f"Error: shorter than random crop length {random_crop_seconds}")
            return None
    elif crop_seconds:
        print(f'cropping {crop_seconds}, waveform shape {waveform.shape}')
        start_crop, end_crop = crop_seconds
        start = int(start_crop * resample[1])
        end = int(end_crop * resample[1])
        if end > waveform.shape[1]:
            print(f"Error: end crop {end_crop} is longer than waveform")
            return None
        waveform = waveform[:, start:end]
        print(f'cropped waveform shape {waveform.shape}')
    if rms_normalised:
        rms = torch.sqrt(torch.mean(torch.square(waveform)))
        waveform = waveform*rms_normalised/rms
    if set_db:
        power = torch.mean(torch.square(waveform))
        normalising_factor = torch.sqrt(10 ** (set_db / 10) / power)
        waveform = waveform * normalising_factor
    if add_white_noise:
        waveform = add_noise_to_waveform(waveform, add_white_noise, 'white')
    if add_pink_noise:
        waveform = add_noise_to_waveform(waveform, add_pink_noise, 'pink')
    if add_brown_noise:
        waveform = add_noise_to_waveform(waveform, add_brown_noise, 'brown')

    if to_spec:
        if to_spec=='power':
            spec = torchaudio.transforms.Spectrogram(
                n_fft=specconfig['n_fft'],
                win_length=specconfig['win_length'], 
                hop_length=specconfig['hop_length'], 
                power=2.0
            )(waveform)
        elif to_spec=='energy':
            spec = torchaudio.transforms.Spectrogram(
                n_fft=specconfig['n_fft'],
                win_length=specconfig['win_length'], 
                hop_length=specconfig['hop_length'], 
                power=None,
            )(waveform)
        return spec

    return waveform

def load_waveform(paths):
    waveforms=[]
    original_sample_rates=[]
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        waveform, original_sample_rate = torchaudio.load(path)
        waveforms.append(waveform)
        original_sample_rates.append(original_sample_rate)
    if len(waveforms)==1:
        return waveforms[0], original_sample_rates[0]
    return waveforms, original_sample_rates