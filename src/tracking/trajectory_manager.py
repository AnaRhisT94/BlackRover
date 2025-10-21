"""Trajectory management for tracking person movement patterns."""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TrackData:
    """Data associated with a single track."""
    track_id: int
    bboxes: List[np.ndarray] = field(default_factory=list)  # [x1, y1, x2, y2]
    centroids: List[np.ndarray] = field(default_factory=list)  # [x, y]
    keypoints: List[np.ndarray] = field(default_factory=list)  # [17, 2]
    keypoint_confs: List[np.ndarray] = field(default_factory=list)  # [17]
    wrist_positions: List[Tuple[Optional[Tuple], Optional[Tuple]]] = field(default_factory=list)  # [(left), (right)]
    frame_indices: List[int] = field(default_factory=list)
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None
    
    def __post_init__(self):
        """Initialize first_seen if not set."""
        if self.first_seen is None and len(self.frame_indices) > 0:
            self.first_seen = self.frame_indices[0]
        if self.last_seen is None and len(self.frame_indices) > 0:
            self.last_seen = self.frame_indices[-1]


class TrajectoryManager:
    """Manage trajectories and historical data for all tracked persons."""
    
    def __init__(self, max_history: int = 300):
        """
        Initialize trajectory manager.
        
        Args:
            max_history: Maximum number of frames to store per track
        """
        self.max_history = max_history
        self.tracks: Dict[int, TrackData] = {}
        
    def update(
        self,
        frame_idx: int,
        track_ids: np.ndarray,
        track_boxes: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        keypoint_confs: Optional[np.ndarray] = None,
        wrist_positions: Optional[List[Tuple[Optional[Tuple], Optional[Tuple]]]] = None
    ):
        """
        Update trajectory data for current frame.
        
        Args:
            frame_idx: Current frame index
            track_ids: Array of track IDs [N]
            track_boxes: Array of bboxes [N, 4]
            keypoints: Optional array of keypoints [N, 17, 2]
            keypoint_confs: Optional array of keypoint confidences [N, 17]
            wrist_positions: Optional list of (left_wrist, right_wrist) tuples
        """
        for i, track_id in enumerate(track_ids):
            # Initialize track if new
            if track_id not in self.tracks:
                self.tracks[track_id] = TrackData(
                    track_id=track_id,
                    first_seen=frame_idx
                )
            
            track = self.tracks[track_id]
            
            # Update bbox
            bbox = track_boxes[i]
            track.bboxes.append(bbox)
            
            # Calculate and store centroid
            centroid = np.array([
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            ])
            track.centroids.append(centroid)
            
            # Store keypoints if available
            if keypoints is not None and i < len(keypoints):
                track.keypoints.append(keypoints[i])
            
            # Store keypoint confidences if available
            if keypoint_confs is not None and i < len(keypoint_confs):
                track.keypoint_confs.append(keypoint_confs[i])
            
            # Store wrist positions if available
            if wrist_positions is not None and i < len(wrist_positions):
                track.wrist_positions.append(wrist_positions[i])
            
            # Update frame index
            track.frame_indices.append(frame_idx)
            track.last_seen = frame_idx
            
            # Trim history if too long
            if len(track.bboxes) > self.max_history:
                track.bboxes = track.bboxes[-self.max_history:]
                track.centroids = track.centroids[-self.max_history:]
                track.keypoints = track.keypoints[-self.max_history:]
                track.keypoint_confs = track.keypoint_confs[-self.max_history:]
                track.wrist_positions = track.wrist_positions[-self.max_history:]
                track.frame_indices = track.frame_indices[-self.max_history:]
    
    def get_track(self, track_id: int) -> Optional[TrackData]:
        """Get track data by ID."""
        return self.tracks.get(track_id)
    
    def get_trajectory(self, track_id: int) -> Optional[np.ndarray]:
        """Get centroid trajectory for a track."""
        track = self.get_track(track_id)
        if track is None or not track.centroids:
            return None
        return np.array(track.centroids)
    
    def get_velocity(self, track_id: int, window: int = 5) -> Optional[float]:
        """
        Calculate average velocity over recent frames.
        
        Args:
            track_id: Track ID
            window: Number of frames to average over
        
        Returns:
            Average velocity in pixels/frame or None
        """
        trajectory = self.get_trajectory(track_id)
        if trajectory is None or len(trajectory) < 2:
            return None
        
        recent = trajectory[-window:] if len(trajectory) >= window else trajectory
        if len(recent) < 2:
            return None
        
        # Calculate displacements between consecutive frames
        displacements = np.linalg.norm(np.diff(recent, axis=0), axis=1)
        return np.mean(displacements)
    
    def get_wrist_velocity(
        self,
        track_id: int,
        wrist_side: str = 'both',
        window: int = 5
    ) -> Optional[float]:
        """
        Calculate average wrist velocity.
        
        Args:
            track_id: Track ID
            wrist_side: 'left', 'right', or 'both'
            window: Number of frames to average over
        
        Returns:
            Average wrist velocity in pixels/frame or None
        """
        track = self.get_track(track_id)
        if track is None or not track.wrist_positions:
            return None
        
        recent_wrists = track.wrist_positions[-window:] if len(track.wrist_positions) >= window else track.wrist_positions
        
        # Extract positions
        positions = []
        for left_wrist, right_wrist in recent_wrists:
            if wrist_side == 'left' and left_wrist is not None:
                positions.append(np.array(left_wrist))
            elif wrist_side == 'right' and right_wrist is not None:
                positions.append(np.array(right_wrist))
            elif wrist_side == 'both':
                if left_wrist is not None and right_wrist is not None:
                    # Use average of both wrists
                    avg_pos = (np.array(left_wrist) + np.array(right_wrist)) / 2
                    positions.append(avg_pos)
                elif left_wrist is not None:
                    positions.append(np.array(left_wrist))
                elif right_wrist is not None:
                    positions.append(np.array(right_wrist))
        
        if len(positions) < 2:
            return None
        
        positions = np.array(positions)
        displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return np.mean(displacements)
    
    def get_active_tracks(self, current_frame: int, max_age: int = 30) -> List[int]:
        """
        Get list of active track IDs.
        
        Args:
            current_frame: Current frame index
            max_age: Maximum frames since last seen
        
        Returns:
            List of active track IDs
        """
        active = []
        for track_id, track in self.tracks.items():
            if track.last_seen is not None and (current_frame - track.last_seen) <= max_age:
                active.append(track_id)
        return active
    
    def get_track_duration(self, track_id: int) -> Optional[int]:
        """Get duration of track in frames."""
        track = self.get_track(track_id)
        if track is None or track.first_seen is None or track.last_seen is None:
            return None
        return track.last_seen - track.first_seen + 1
    
    def clear_old_tracks(self, current_frame: int, max_age: int = 300):
        """
        Remove tracks that haven't been seen recently.
        
        Args:
            current_frame: Current frame index
            max_age: Maximum frames since last seen before removal
        """
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.last_seen is not None and (current_frame - track.last_seen) > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def get_statistics(self) -> Dict:
        """Get statistics about tracked persons."""
        if not self.tracks:
            return {
                'total_tracks': 0,
                'avg_duration': 0,
                'max_duration': 0,
                'min_duration': 0
            }
        
        durations = [self.get_track_duration(tid) for tid in self.tracks.keys()]
        durations = [d for d in durations if d is not None]
        
        return {
            'total_tracks': len(self.tracks),
            'avg_duration': np.mean(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0
        }
    
    def __repr__(self):
        return f"TrajectoryManager(tracks={len(self.tracks)}, max_history={self.max_history})"

