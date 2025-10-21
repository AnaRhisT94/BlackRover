"""YOLO11-Pose detector for keypoint extraction."""
from ultralytics import YOLO
import numpy as np
from typing import Tuple, List, Optional


class PoseDetector:
    """YOLO11 pose estimation detector."""
    
    # COCO keypoint indices
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    KEYPOINT_INDICES = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
    
    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.5,
        keypoint_threshold: float = 0.3,
        device: str = "cuda"
    ):
        """
        Initialize pose detector.
        
        Args:
            model_path: Path to YOLO11-pose model
            conf_threshold: Confidence threshold for person detection
            keypoint_threshold: Confidence threshold for keypoints
            device: Device to run inference on
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.keypoint_threshold = keypoint_threshold
        self.device = device
        
        # Load model
        self.model = YOLO(model_path)
        if device == "cuda":
            self.model.to('cuda')
    
    def detect_poses(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect poses in a frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Tuple of:
                - keypoints: Array of shape [N, 17, 2] (x, y coordinates)
                - keypoint_confidences: Array of shape [N, 17]
                - boxes: Array of shape [N, 4] (x1, y1, x2, y2)
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        if results.keypoints is None or len(results.keypoints) == 0 or results.keypoints.conf is None:
            return np.array([]), np.array([]), np.array([])
        
        # Extract keypoints (x, y coordinates)
        keypoints = results.keypoints.xy.cpu().numpy()  # Shape: [N, 17, 2]
        
        # Extract keypoint confidences
        keypoint_conf = results.keypoints.conf.cpu().numpy()  # Shape: [N, 17]
        
        # Extract bounding boxes
        boxes = results.boxes.xyxy.cpu().numpy()  # Shape: [N, 4]
        
        return keypoints, keypoint_conf, boxes
    
    def get_wrist_positions(
        self,
        keypoints: np.ndarray,
        keypoint_conf: np.ndarray
    ) -> Tuple[List[Optional[Tuple[float, float]]], List[Optional[Tuple[float, float]]]]:
        """
        Extract wrist positions from keypoints.
        
        Args:
            keypoints: Keypoints array [N, 17, 2]
            keypoint_conf: Confidence array [N, 17]
        
        Returns:
            Tuple of (left_wrists, right_wrists) lists, each containing (x, y) or None
        """
        left_wrist_idx = self.KEYPOINT_INDICES["left_wrist"]
        right_wrist_idx = self.KEYPOINT_INDICES["right_wrist"]
        
        left_wrists = []
        right_wrists = []
        
        for i in range(len(keypoints)):
            # Left wrist
            if keypoint_conf[i, left_wrist_idx] >= self.keypoint_threshold:
                left_wrists.append(tuple(keypoints[i, left_wrist_idx]))
            else:
                left_wrists.append(None)
            
            # Right wrist
            if keypoint_conf[i, right_wrist_idx] >= self.keypoint_threshold:
                right_wrists.append(tuple(keypoints[i, right_wrist_idx]))
            else:
                right_wrists.append(None)
        
        return left_wrists, right_wrists

