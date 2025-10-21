"""Geometry utility functions for bbox and point operations."""
import numpy as np
from typing import Tuple, List


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU score (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def point_in_bbox(point: Tuple[float, float], bbox: np.ndarray) -> bool:
    """
    Check if a point is inside a bounding box.
    
    Args:
        point: (x, y) coordinates
        bbox: [x1, y1, x2, y2]
    
    Returns:
        True if point is inside bbox
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def expand_bbox(bbox: np.ndarray, expand_ratio: float = 0.1) -> np.ndarray:
    """
    Expand bounding box by a ratio.
    
    Args:
        bbox: [x1, y1, x2, y2]
        expand_ratio: Ratio to expand (0.1 = 10%)
    
    Returns:
        Expanded bbox [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    dw = w * expand_ratio / 2
    dh = h * expand_ratio / 2
    
    return np.array([
        max(0, x1 - dw),
        max(0, y1 - dh),
        x2 + dw,
        y2 + dh
    ])


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def bbox_center(bbox: np.ndarray) -> Tuple[float, float]:
    """Get center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

