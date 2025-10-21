"""Video ingestion and frame extraction."""
import cv2
import numpy as np
from typing import Generator, Tuple, Optional
from pathlib import Path


class VideoReader:
    """Read and process video frames."""
    
    def __init__(self, video_path: str, skip_frames: int = 0):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to input video file
            skip_frames: Number of frames to skip between reads (0 = read all)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        self.skip_frames = skip_frames
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields video frames.
        
        Yields:
            Tuple of (frame_index, frame_array)
        """
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % (self.skip_frames + 1) == 0:
                yield frame_idx, frame
            
            frame_idx += 1
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number.
        
        Args:
            frame_number: Frame index to retrieve
        
        Returns:
            Frame array or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def __repr__(self):
        return (f"VideoReader(path={self.video_path}, "
                f"fps={self.fps:.2f}, "
                f"frames={self.total_frames}, "
                f"resolution={self.width}x{self.height})")

