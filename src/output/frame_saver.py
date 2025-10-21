"""Frame saver utility for saving individual frames as images."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class FrameSaver:
    """Save video frames as individual image files."""
    
    def __init__(self, output_dir: str, prefix: str = "frame", image_format: str = "jpg"):
        """
        Initialize frame saver.
        
        Args:
            output_dir: Directory to save frames
            prefix: Prefix for frame filenames
            image_format: Image format (jpg, png, etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.image_format = image_format
        self.frame_count = 0
        
    def save_frame(self, frame: np.ndarray, frame_idx: Optional[int] = None):
        """
        Save a single frame.
        
        Args:
            frame: Frame to save (BGR format)
            frame_idx: Optional frame index (uses internal counter if None)
        """
        if frame_idx is None:
            frame_idx = self.frame_count
            
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Generate filename with zero-padding
        filename = f"{self.prefix}_{frame_idx:06d}.{self.image_format}"
        filepath = self.output_dir / filename
        
        # Save frame
        cv2.imwrite(str(filepath), frame)
        self.frame_count += 1
        
        return filepath
    
    def get_saved_count(self) -> int:
        """Get number of frames saved."""
        return self.frame_count
    
    def __repr__(self):
        return f"FrameSaver(dir={self.output_dir}, saved={self.frame_count})"

