"""Visualization utilities for drawing on frames."""
import cv2
import numpy as np
import supervision as sv
from typing import Optional, List, Tuple


class Visualizer:
    """Draw annotations on video frames."""
    
    # Color palette (BGR format)
    COLORS = {
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }
    
    # Pose skeleton connections (COCO format)
    SKELETON = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    def __init__(self):
        """Initialize visualizer."""
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            color_lookup=sv.ColorLookup.TRACK  # Use track ID for consistent colors
        )
        self.label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.TRACK  # Match box annotator color lookup
        )
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes for detections.
        
        Args:
            frame: Input frame
            detections: Supervision Detections object
            labels: Optional list of labels for each detection
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        annotated = self.box_annotator.annotate(annotated, detections)
        
        if labels:
            annotated = self.label_annotator.annotate(annotated, detections, labels)
        
        return annotated
    
    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        threshold: float = 0.3
    ) -> np.ndarray:
        """
        Draw pose keypoints and skeleton.
        
        Args:
            frame: Input frame
            keypoints: Array of shape [N, 17, 2]
            confidences: Optional confidence scores [N, 17]
            threshold: Confidence threshold to draw keypoints
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for person_idx in range(len(keypoints)):
            kpts = keypoints[person_idx]
            confs = confidences[person_idx] if confidences is not None else np.ones(17)
            
            # Draw skeleton connections
            for start_idx, end_idx in self.SKELETON:
                if confs[start_idx] >= threshold and confs[end_idx] >= threshold:
                    start_point = tuple(kpts[start_idx].astype(int))
                    end_point = tuple(kpts[end_idx].astype(int))
                    cv2.line(annotated, start_point, end_point, self.COLORS['green'], 2)
            
            # Draw keypoints
            for kpt_idx in range(len(kpts)):
                if confs[kpt_idx] >= threshold:
                    point = tuple(kpts[kpt_idx].astype(int))
                    cv2.circle(annotated, point, 4, self.COLORS['red'], -1)
        
        return annotated
    
    def draw_wrists(
        self,
        frame: np.ndarray,
        left_wrist: Optional[Tuple[float, float]],
        right_wrist: Optional[Tuple[float, float]],
        radius: int = 8,
        color: tuple = None
    ) -> np.ndarray:
        """
        Draw wrist positions.
        
        Args:
            frame: Input frame
            left_wrist: (x, y) position or None
            right_wrist: (x, y) position or None
            radius: Circle radius
            color: BGR color tuple
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        color = color or self.COLORS['yellow']
        
        if left_wrist:
            cv2.circle(annotated, tuple(map(int, left_wrist)), radius, color, -1)
            cv2.putText(annotated, "L", tuple(map(int, left_wrist)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 2)
        
        if right_wrist:
            cv2.circle(annotated, tuple(map(int, right_wrist)), radius, color, -1)
            cv2.putText(annotated, "R", tuple(map(int, right_wrist)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 2)
        
        return annotated
    
    def draw_alert(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        message: str,
        alert_level: str = "SUSPICIOUS"
    ) -> np.ndarray:
        """
        Draw alert on frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            message: Alert message
            alert_level: "NORMAL", "SUSPICIOUS", or "ALERT"
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Choose color based on alert level
        color_map = {
            "NORMAL": self.COLORS['green'],
            "SUSPICIOUS": self.COLORS['yellow'],
            "ALERT": self.COLORS['red']
        }
        color = color_map.get(alert_level, self.COLORS['red'])
        
        # Draw thick box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label = f"{alert_level}: {message}"
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 10, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.COLORS['white'],
            2
        )
        
        return annotated
    
    def get_torso_bbox(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        Get torso bounding box from keypoints.
        
        Args:
            keypoints: Single person keypoints [17, 2]
        
        Returns:
            Torso bbox [x1, y1, x2, y2] or None
        """
        # Use shoulders and hips to define torso
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
        left_hip, right_hip = keypoints[11], keypoints[12]
        
        # Check if all keypoints are valid (non-zero)
        torso_points = [left_shoulder, right_shoulder, left_hip, right_hip]
        if any(np.all(pt == 0) for pt in torso_points):
            return None
        
        all_points = np.array(torso_points)
        x1, y1 = all_points.min(axis=0)
        x2, y2 = all_points.max(axis=0)
        
        # Expand slightly
        margin = 0.1
        w, h = x2 - x1, y2 - y1
        x1 -= w * margin
        y1 -= h * margin
        x2 += w * margin
        y2 += h * margin
        
        return np.array([x1, y1, x2, y2])

