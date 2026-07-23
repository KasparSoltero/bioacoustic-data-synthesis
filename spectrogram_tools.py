import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from matplotlib.colors import hsv_to_rgb

def pcen(spec, s=0.025, alpha=0.01, delta=0, r=0.05, eps=1e-6):
    """
    Apply Per-Channel Energy Normalization (PCEN) to a spectrogram.
    Uses a robust 1D convolution to implement the moving average filter.
    """
    try:
        if spec.ndim not in [3, 4]:
            raise ValueError(f"Input tensor must be either 3D or 4D, but got {spec.ndim}D tensor.")

        device = spec.device
        orig_shape = spec.shape
        
        if spec.ndim == 4:
            spec = spec.view(-1, orig_shape[-2], orig_shape[-1])
            
        in_channels = spec.shape[1] 
        time_steps = spec.shape[2]
        kernel_size = max(1, int(s * time_steps))

        ma_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding='same', 
            bias=False,
            groups=in_channels
        ).to(device)

        ma_conv.weight.data.fill_(1.0 / kernel_size)
        ma_conv.weight.requires_grad = False 

        M = ma_conv(spec)
        pcen_spec = (spec / (M + eps).pow(alpha) + delta).pow(r) - delta**r

        if len(orig_shape) == 4:
            pcen_spec = pcen_spec.view(orig_shape)

    except Exception as e:
        print(f"Error during PCEN processing: {e}")
        print("PCEN failed, returning original spectrogram.")
        return spec.view(orig_shape)
    
    return pcen_spec

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
            spec = torch.square(spec)
            spec = pcen(spec)

    spec = np.squeeze(spec.numpy())
    spec = np.flipud(spec)
    spec = (spec - spec.min()) / (spec.max() - spec.min()) 

    if color_mode == 'HSV':
        value = spec
        saturation = 4 * value * (1 - value)
        hue = np.linspace(0,1,spec.shape[0])[:, np.newaxis] 
        hue = np.tile(hue, (1, spec.shape[1]))
        hsv_spec = np.stack([hue, saturation, value], axis=-1)
        rgb_spec = hsv_to_rgb(hsv_spec)
        rgb_spec = np.clip(rgb_spec, 0, 1)
        spec = Image.fromarray(np.uint8(rgb_spec * 255), 'RGB')
    elif color_mode == 'RGB':
        spec = np.stack([spec, spec, spec], axis=-1)
        spec = Image.fromarray(np.uint8(spec * 255), 'RGB')
    else:
        spec = Image.fromarray(np.uint8(spec * 255), 'L')

    if resize:
        spec = spec.resize(resize, Image.Resampling.LANCZOS)
    
    return spec

def resample_log_mask_to_linear(log_space_mask, linear_spec_shape, log_base=10.0):
    """
    Resamples a mask from a logarithmic frequency scale to a linear frequency scale.
    """
    linear_height, linear_width = linear_spec_shape
    
    y = torch.linspace(-1, 1, linear_height, device=log_space_mask.device)
    x = torch.linspace(-1, 1, linear_width, device=log_space_mask.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    y_norm = (grid_y + 1) / 2
    
    # Inverse map log back to linear space dependent on parameterised base
    log_y_norm = torch.log(y_norm * (log_base - 1) + 1) / torch.log(torch.tensor(log_base, dtype=torch.float32, device=y_norm.device))
    log_grid_y = log_y_norm * 2 - 1

    target_grid = torch.stack((grid_x, log_grid_y), dim=-1).unsqueeze(0)
    log_space_mask_unsqueezed = log_space_mask.float().unsqueeze(0).unsqueeze(0)
    
    linear_mask = F.grid_sample(
        log_space_mask_unsqueezed, 
        target_grid, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=False
    )
    
    return linear_mask.squeeze(0).squeeze(0)

def map_frequency_to_log_scale(original_height, freq_indices, log_base=10.0):
    log_freq_indices = []
    for freq_index in freq_indices:
        relative_position = freq_index / (original_height - 1 if original_height > 1 else 1)
        log_position = torch.log(torch.tensor(relative_position * (log_base - 1) + 1)) / torch.log(torch.tensor(log_base, dtype=torch.float32))
        log_index = int(torch.round(log_position * (original_height - 1)))
        log_freq_indices.append(log_index)
    return log_freq_indices

def map_frequency_to_linear_scale(original_height, freq_indices, log_base=10.0):
    linear_freq_indices = []
    for freq_index in freq_indices:
        relative_position = freq_index / (original_height - 1 if original_height > 1 else 1)
        linear_position = (log_base ** relative_position - 1) / (log_base - 1)
        linear_index = int(torch.round(linear_position * (original_height - 1)))
        linear_freq_indices.append(linear_index)
    return linear_freq_indices

# --- Bounding Box Utilities ---

def calculate_iou_ios(box1, box2, format):
    if format == 'xxyy':
        x_left = max(box1[0], box2[0])
        y_top = max(box1[2], box2[2])
        x_right = min(box1[1], box2[1])
        y_bottom = min(box1[3], box2[3])
        box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
        box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
        intersection_area = (x_right-x_left) * (y_bottom-y_top)
        if x_right < x_left or y_bottom < y_top:
            return 0.0, 0
    elif format == 'xyxy':
        x_left = max(box1[0], box2[0])
        y_bottom = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_top = min(box1[3], box2[3])
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        intersection_area = (x_right-x_left) * (y_top-y_bottom)
        if x_right < x_left or y_top < y_bottom or box1_area == 0 or box2_area == 0:
            return 0.0, 0

    ios = intersection_area / min(box1_area, box2_area)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou, ios

def combine_boxes(box1, box2, format='xxyy'):
    if format == 'xxyy':
        x_min = min(box1[0], box2[0])
        x_max = max(box1[1], box2[1])
        y_min = min(box1[2], box2[2])
        y_max = max(box1[3], box2[3])
        return [x_min, x_max, y_min, y_max]
    elif format == 'xyxy':
        x_min = min(box1[0], box2[0])
        x_max = max(box1[2], box2[2])
        y_min = min(box1[1], box2[1])
        y_max = max(box1[3], box2[3])
        return [x_min, y_min, x_max, y_max]

def find(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i]) 
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
    
    updated = True
    while updated:
        updated = False
        temp_merged_boxes = merged_boxes.copy()
        for root1 in list(temp_merged_boxes.keys()):
            for root2 in list(temp_merged_boxes.keys()):
                if root1 == root2:
                    continue
                if classes[root1] != classes[root2]: 
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

    final_boxes = list(merged_boxes.values())
    final_classes = [classes[root] for root in merged_boxes]
    
    return final_boxes, final_classes