# bioacoustic_synthesis/annotations.py
# Bounding-box geometry and class-wise merging for synthesised annotations.

from typing import List, Sequence, Tuple


def calculate_iou_ios(box1, box2, format: str = 'xxyy') -> Tuple[float, float]:
    """Returns (IoU, IoS) — intersection over union, and over the smaller box."""
    if format == 'xxyy':
        (a_x0, a_x1, a_y0, a_y1) = box1
        (b_x0, b_x1, b_y0, b_y1) = box2
    elif format == 'xyxy':
        (a_x0, a_y0, a_x1, a_y1) = box1
        (b_x0, b_y0, b_x1, b_y1) = box2
    else:
        raise ValueError(f"Unknown box format: {format}")

    inter_w = min(a_x1, b_x1) - max(a_x0, b_x0)
    inter_h = min(a_y1, b_y1) - max(a_y0, b_y0)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0, 0.0

    inter = inter_w * inter_h
    area_a = (a_x1 - a_x0) * (a_y1 - a_y0)
    area_b = (b_x1 - b_x0) * (b_y1 - b_y0)
    if area_a <= 0 or area_b <= 0:
        return 0.0, 0.0

    return inter / (area_a + area_b - inter), inter / min(area_a, area_b)


def combine_boxes(box1, box2, format: str = 'xxyy') -> List[float]:
    """Returns the smallest box enclosing both inputs."""
    if format == 'xxyy':
        return [min(box1[0], box2[0]), max(box1[1], box2[1]),
                min(box1[2], box2[2]), max(box1[3], box2[3])]
    elif format == 'xyxy':
        return [min(box1[0], box2[0]), min(box1[1], box2[1]),
                max(box1[2], box2[2]), max(box1[3], box2[3])]
    raise ValueError(f"Unknown box format: {format}")


def _find(parent: List[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent: List[int], rank: List[int], i: int, j: int):
    ri, rj = _find(parent, i), _find(parent, j)
    if ri == rj:
        return
    if rank[ri] < rank[rj]:
        ri, rj = rj, ri
    parent[rj] = ri
    if rank[ri] == rank[rj]:
        rank[ri] += 1


def merge_boxes_by_class(
    boxes: Sequence,
    classes: Sequence[int],
    iou_threshold: float = 0.5,
    ios_threshold: float = 0.5,
    format: str = 'xxyy',
):
    """
    Merges same-class boxes that overlap beyond either threshold. Union-find
    groups the initial pairs, then merged groups are re-tested against each
    other until no further merges occur (a merged box can newly overlap a
    third box that neither original did).
    """
    if not boxes:
        return [], []

    parent = list(range(len(boxes)))
    rank = [0] * len(boxes)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if classes[i] != classes[j]:
                continue
            iou, ios = calculate_iou_ios(boxes[i], boxes[j], format)
            if iou > iou_threshold or ios > ios_threshold:
                _union(parent, rank, i, j)

    groups = {}
    for i in range(len(boxes)):
        root = _find(parent, i)
        groups[root] = combine_boxes(groups[root], boxes[i], format) if root in groups else list(boxes[i])

    merged = [(box, classes[root]) for root, box in groups.items()]

    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if merged[i][1] != merged[j][1]:
                    continue
                iou, ios = calculate_iou_ios(merged[i][0], merged[j][0], format)
                if iou > iou_threshold or ios > ios_threshold:
                    merged[i] = (combine_boxes(merged[i][0], merged[j][0], format), merged[i][1])
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break

    return [b for b, _ in merged], [c for _, c in merged]