"""Video output writer."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class AnnotatedVideoWriter:
    """Write annotated video frames to output file."""
    
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Optional[tuple] = None,
        codec: str = 'mp4v'
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: (width, height), if None will be set on first frame
            codec: FourCC codec code
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.frame_size = frame_size
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.writer = None
        self.frame_count = 0
    
    def write(self, frame: np.ndarray):
        """
        Write a frame to video.
        
        Args:
            frame: Frame to write (BGR format)
        """
        if self.writer is None:
            # Initialize writer on first frame
            if self.frame_size is None:
                h, w = frame.shape[:2]
                self.frame_size = (w, h)
            
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                self.codec,
                self.fps,
                self.frame_size
            )
        
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        """Release video writer resources."""
        if self.writer:
            self.writer.release()
            print(f"Video saved: {self.output_path} ({self.frame_count} frames)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

