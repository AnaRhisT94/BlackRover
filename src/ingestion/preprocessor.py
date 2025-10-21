"""Frame preprocessing utilities."""
import cv2
import numpy as np
from typing import Tuple, Optional


class FramePreprocessor:
    """Preprocess video frames before detection."""
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        auto_orient: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Resize frames to (width, height), None to keep original
            normalize: Normalize pixel values to [0, 1]
            auto_orient: Auto-rotate frames based on EXIF data
        """
        self.target_size = target_size
        self.normalize = normalize
        self.auto_orient = auto_orient
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Processed frame
        """
        processed = frame.copy()
        
        # Resize if needed
        if self.target_size:
            processed = cv2.resize(
                processed,
                self.target_size,
                interpolation=cv2.INTER_LINEAR
            )
        
        # Normalize if needed
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    @staticmethod
    def denormalize(frame: np.ndarray) -> np.ndarray:
        """Convert normalized frame back to uint8."""
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            return (frame * 255).astype(np.uint8)
        return frame

