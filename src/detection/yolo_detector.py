"""YOLO-based object and person detection."""
from ultralytics import YOLO
import supervision as sv
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class YOLODetector:
    """YOLO11 object detector."""
    
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda"
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model
        self.model = YOLO(model_path)
        if device == "cuda":
            self.model.to('cuda')
    
    def detect_objects(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> sv.Detections:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame (BGR format)
            classes: List of class IDs to detect (None = all classes)
        
        Returns:
            Supervision Detections object
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        return detections
    
    def detect_persons(self, frame: np.ndarray) -> sv.Detections:
        """Detect only persons (class 0 in COCO)."""
        return self.detect_objects(frame, classes=[0])

