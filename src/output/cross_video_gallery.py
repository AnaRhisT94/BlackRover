"""Cross-video gallery for visualizing person tracks across multiple videos."""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class TrackSnapshot:
    """Represents a snapshot of a tracked person."""
    track_id: int
    global_id: int
    video_name: str
    frame_idx: int
    bbox: np.ndarray
    confidence: float
    crop: Optional[np.ndarray] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'track_id': int(self.track_id),
            'global_id': int(self.global_id),
            'video_name': self.video_name,
            'frame_idx': int(self.frame_idx),
            'bbox': self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox,
            'confidence': float(self.confidence)
        }


class CrossVideoGalleryVisualizer:
    """
    Visualizes person tracks across multiple videos with global ID assignments.
    
    Creates a grid showing:
    - All videos processed
    - All tracks in each video
    - Global ID assigned to each track
    """
    
    def __init__(self, output_dir: str, crop_size: Tuple[int, int] = (128, 256)):
        """
        Initialize cross-video gallery visualizer.
        
        Args:
            output_dir: Directory to save gallery output
            crop_size: Size to resize crops to (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crop_size = crop_size
        
        # Store track snapshots organized by video
        self.videos: Dict[str, List[TrackSnapshot]] = {}
        self.video_order: List[str] = []  # Preserve order of videos processed
        
    def add_track(
        self,
        track_id: int,
        global_id: int,
        video_name: str,
        frame_idx: int,
        bbox: np.ndarray,
        frame: np.ndarray,
        confidence: float
    ):
        """
        Add or update a track snapshot.
        
        Args:
            track_id: Local track ID in the video
            global_id: Global ReID ID assigned
            video_name: Name of the video
            frame_idx: Frame index
            bbox: Bounding box [x1, y1, x2, y2]
            frame: Full frame image
            confidence: Detection confidence
        """
        # Initialize video entry if needed
        if video_name not in self.videos:
            self.videos[video_name] = []
            self.video_order.append(video_name)
        
        # Extract crop from frame
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, self.crop_size)
        else:
            crop = None
        
        # Check if we already have a snapshot for this track_id in this video
        existing_idx = None
        for i, snapshot in enumerate(self.videos[video_name]):
            if snapshot.track_id == track_id:
                existing_idx = i
                break
        
        # Create new snapshot
        snapshot = TrackSnapshot(
            track_id=track_id,
            global_id=global_id,
            video_name=video_name,
            frame_idx=frame_idx,
            bbox=bbox,
            confidence=confidence,
            crop=crop
        )
        
        if existing_idx is not None:
            # Update if new confidence is higher
            if confidence > self.videos[video_name][existing_idx].confidence:
                self.videos[video_name][existing_idx] = snapshot
        else:
            # Add new track
            self.videos[video_name].append(snapshot)
    
    def generate_grid(
        self,
        output_path: Optional[str] = None,
        max_cols: int = 8,
        padding: int = 10,
        font_scale: float = 0.6,
        font_thickness: int = 2
    ) -> str:
        """
        Generate cross-video grid visualization.
        
        Args:
            output_path: Path to save grid image (default: output_dir/cross_video_grid.jpg)
            max_cols: Maximum number of columns per row
            padding: Padding between crops in pixels
            font_scale: Font scale for labels
            font_thickness: Font thickness for labels
        
        Returns:
            Path to saved grid image
        """
        if output_path is None:
            output_path = self.output_dir / "cross_video_grid.jpg"
        else:
            output_path = Path(output_path)
        
        if not self.videos:
            print("No tracks to visualize!")
            return str(output_path)
        
        # Calculate dimensions
        crop_width, crop_height = self.crop_size
        label_height = 40  # Height for ID label
        video_header_height = 60  # Height for video name header
        total_crop_height = crop_height + label_height
        
        # Build grid sections for each video
        video_sections = []
        max_width = 0
        
        for video_name in self.video_order:
            tracks = self.videos[video_name]
            if not tracks:
                continue
            
            # Sort tracks by global_id for consistent display
            tracks_sorted = sorted(tracks, key=lambda t: t.global_id)
            
            # Calculate rows and columns for this video
            num_tracks = len(tracks_sorted)
            num_cols = min(max_cols, num_tracks)
            num_rows = (num_tracks + num_cols - 1) // num_cols
            
            # Create video section
            section_width = num_cols * (crop_width + padding) + padding
            section_height = video_header_height + num_rows * (total_crop_height + padding) + padding
            
            section = np.ones((section_height, section_width, 3), dtype=np.uint8) * 255
            
            # Draw video header
            cv2.rectangle(section, (0, 0), (section_width, video_header_height), (240, 240, 240), -1)
            header_text = f"Video: {video_name}"
            text_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, font_thickness)[0]
            text_x = (section_width - text_size[0]) // 2
            text_y = (video_header_height + text_size[1]) // 2
            cv2.putText(section, header_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 0, 0), font_thickness)
            
            # Add track info
            track_count_text = f"({num_tracks} tracks)"
            track_text_size = cv2.getTextSize(track_count_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, 1)[0]
            track_text_x = (section_width - track_text_size[0]) // 2
            track_text_y = text_y + text_size[1] + 5
            cv2.putText(section, track_count_text, (track_text_x, track_text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (100, 100, 100), 1)
            
            # Place crops in grid
            for idx, track in enumerate(tracks_sorted):
                row = idx // num_cols
                col = idx % num_cols
                
                x = padding + col * (crop_width + padding)
                y = video_header_height + padding + row * (total_crop_height + padding)
                
                # Draw crop if available
                if track.crop is not None:
                    section[y:y+crop_height, x:x+crop_width] = track.crop
                else:
                    # Draw placeholder
                    cv2.rectangle(section, (x, y), (x+crop_width, y+crop_height), (200, 200, 200), -1)
                    cv2.putText(section, "N/A", (x+20, y+crop_height//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                
                # Draw ID label below crop
                label_y = y + crop_height
                cv2.rectangle(section, (x, label_y), (x+crop_width, label_y+label_height), (50, 50, 50), -1)
                
                # Global ID label
                id_text = f"ID: {track.global_id}"
                id_text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                id_text_x = x + (crop_width - id_text_size[0]) // 2
                id_text_y = label_y + (label_height + id_text_size[1]) // 2
                cv2.putText(section, id_text, (id_text_x, id_text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                
                # Track ID (smaller, below global ID)
                track_text = f"(track {track.track_id})"
                track_text_size = cv2.getTextSize(track_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.5, 1)[0]
                track_text_x = x + (crop_width - track_text_size[0]) // 2
                track_text_y = id_text_y + id_text_size[1] + 3
                cv2.putText(section, track_text, (track_text_x, track_text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.5, (180, 180, 180), 1)
            
            video_sections.append(section)
            max_width = max(max_width, section_width)
        
        # Combine all video sections vertically
        total_height = sum(s.shape[0] for s in video_sections) + padding * (len(video_sections) + 1)
        
        final_grid = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255
        
        current_y = padding
        for section in video_sections:
            h, w = section.shape[:2]
            x_offset = (max_width - w) // 2
            final_grid[current_y:current_y+h, x_offset:x_offset+w] = section
            current_y += h + padding
        
        # Save grid
        cv2.imwrite(str(output_path), final_grid)
        print(f"Cross-video grid saved to: {output_path}")
        
        return str(output_path)
    
    def save_metadata(self, filepath: Optional[str] = None):
        """Save track metadata to JSON."""
        if filepath is None:
            filepath = self.output_dir / "cross_video_metadata.json"
        else:
            filepath = Path(filepath)
        
        metadata = {
            'total_videos': len(self.videos),
            'videos': {}
        }
        
        for video_name in self.video_order:
            tracks = self.videos[video_name]
            metadata['videos'][video_name] = {
                'total_tracks': len(tracks),
                'tracks': [track.to_dict() for track in tracks]
            }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Cross-video metadata saved to: {filepath}")
        return str(filepath)
    
    def get_statistics(self) -> dict:
        """Get gallery statistics."""
        total_tracks = sum(len(tracks) for tracks in self.videos.values())
        unique_global_ids = set()
        
        for tracks in self.videos.values():
            for track in tracks:
                unique_global_ids.add(track.global_id)
        
        return {
            'total_videos': len(self.videos),
            'total_tracks': total_tracks,
            'unique_persons': len(unique_global_ids),
            'videos': {
                video: len(tracks) for video, tracks in self.videos.items()
            }
        }
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"CrossVideoGalleryVisualizer(videos={stats['total_videos']}, "
                f"tracks={stats['total_tracks']}, unique_persons={stats['unique_persons']})")


