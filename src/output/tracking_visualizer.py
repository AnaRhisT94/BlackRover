"""Enhanced visualization for tracking debugging and analysis."""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..tracking.trajectory_manager import TrajectoryManager


class TrackingVisualizer:
    """Advanced visualizer for debugging tracking behavior."""
    
    # Color palette (BGR format)
    COLORS = {
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'orange': (0, 165, 255),
        'purple': (255, 51, 153)
    }
    
    # Assign distinct colors to track IDs
    TRACK_COLORS = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (0, 165, 255),  # Orange
        (255, 51, 153), # Purple
        (128, 128, 0),  # Teal
        (128, 0, 128),  # Purple-ish
    ]
    
    def __init__(self, trajectory_length: int = 60):
        """
        Initialize tracking visualizer.
        
        Args:
            trajectory_length: Number of trajectory points to display
        """
        self.trajectory_length = trajectory_length
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for a track ID."""
        return self.TRACK_COLORS[track_id % len(self.TRACK_COLORS)]
    
    def draw_track_box(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: np.ndarray,
        thickness: int = 3
    ) -> np.ndarray:
        """
        Draw bounding box for a track with track-specific color.
        
        Args:
            frame: Input frame
            track_id: Track ID
            bbox: Bounding box [x1, y1, x2, y2]
            thickness: Box thickness
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        color = self.get_track_color(track_id)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        return annotated
    
    def draw_track_label(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: np.ndarray,
        velocity: Optional[float] = None,
        duration: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw detailed track label.
        
        Args:
            frame: Input frame
            track_id: Track ID
            bbox: Bounding box [x1, y1, x2, y2]
            velocity: Optional velocity value
            duration: Optional track duration in frames
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        color = self.get_track_color(track_id)
        x1, y1, x2, y2 = map(int, bbox)
        
        # Build label
        label_parts = [f"ID: {track_id}"]
        if velocity is not None:
            label_parts.append(f"V: {velocity:.1f} px/f")
        if duration is not None:
            label_parts.append(f"D: {duration}f")
        
        label = " | ".join(label_parts)
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Position above bbox
        label_y1 = max(y1 - label_h - 15, 0)
        label_y2 = max(y1 - 5, label_h + 10)
        
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + label_w + 10, label_y2),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1 + 5, label_y2 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.COLORS['white'],
            2
        )
        
        return annotated
    
    def draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: np.ndarray,
        track_id: int,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw trajectory path for a track.
        
        Args:
            frame: Input frame
            trajectory: Array of centroids [T, 2]
            track_id: Track ID
            thickness: Line thickness
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        if len(trajectory) < 2:
            return annotated
        
        color = self.get_track_color(track_id)
        
        # Draw lines connecting trajectory points
        points = trajectory[-self.trajectory_length:]
        for i in range(len(points) - 1):
            pt1 = tuple(points[i].astype(int))
            pt2 = tuple(points[i + 1].astype(int))
            
            # Fade older points
            alpha = (i + 1) / len(points)
            curr_thickness = max(1, int(thickness * alpha))
            
            cv2.line(annotated, pt1, pt2, color, curr_thickness)
        
        # Draw current position as a larger circle
        if len(points) > 0:
            current_pos = tuple(points[-1].astype(int))
            cv2.circle(annotated, current_pos, 6, color, -1)
            cv2.circle(annotated, current_pos, 8, self.COLORS['white'], 2)
        
        return annotated
    
    def draw_velocity_arrow(
        self,
        frame: np.ndarray,
        centroid: np.ndarray,
        trajectory: np.ndarray,
        track_id: int,
        scale: float = 3.0
    ) -> np.ndarray:
        """
        Draw velocity vector as an arrow.
        
        Args:
            frame: Input frame
            centroid: Current centroid [x, y]
            trajectory: Recent trajectory points [T, 2]
            track_id: Track ID
            scale: Arrow length scale factor
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        if len(trajectory) < 2:
            return annotated
        
        color = self.get_track_color(track_id)
        
        # Calculate velocity vector from last few points
        recent = trajectory[-5:]
        if len(recent) >= 2:
            velocity = recent[-1] - recent[-2]
            
            # Scale the arrow
            end_point = centroid + velocity * scale
            
            start = tuple(centroid.astype(int))
            end = tuple(end_point.astype(int))
            
            cv2.arrowedLine(annotated, start, end, color, 2, tipLength=0.3)
        
        return annotated
    
    def draw_stats_panel(
        self,
        frame: np.ndarray,
        trajectory_manager: TrajectoryManager,
        current_frame: int,
        active_tracks: int
    ) -> np.ndarray:
        """
        Draw statistics panel with tracking info.
        
        Args:
            frame: Input frame
            trajectory_manager: Trajectory manager instance
            current_frame: Current frame index
            active_tracks: Number of active tracks
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        stats = trajectory_manager.get_statistics()
        
        # Panel settings
        panel_x = 10
        panel_y = 50
        line_height = 25
        
        # Semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (panel_x - 5, panel_y - 25),
            (panel_x + 450, panel_y + line_height * 5 + 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Draw stats
        stats_text = [
            f"Frame: {current_frame}",
            f"Active Tracks: {active_tracks}",
            f"Total Unique IDs: {stats['total_tracks']}",
            f"Avg Track Duration: {stats['avg_duration']:.1f} frames",
            f"Max Track Duration: {stats['max_duration']} frames"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(
                annotated,
                text,
                (panel_x, panel_y + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.COLORS['white'],
                2
            )
        
        return annotated
    
    def draw_track_legend(
        self,
        frame: np.ndarray,
        active_track_ids: List[int],
        position: str = "bottom_right"
    ) -> np.ndarray:
        """
        Draw legend showing active tracks and their colors.
        
        Args:
            frame: Input frame
            active_track_ids: List of active track IDs
            position: Position of legend ("top_right", "bottom_right", etc.)
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Position settings
        if position == "bottom_right":
            start_x = w - 150
            start_y = h - 30 * len(active_track_ids) - 20
        else:  # top_right
            start_x = w - 150
            start_y = 50
        
        # Draw semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (start_x - 10, start_y - 10),
            (w - 10, start_y + 30 * len(active_track_ids)),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        
        # Draw legend entries
        for i, track_id in enumerate(sorted(active_track_ids)):
            y = start_y + i * 30
            color = self.get_track_color(track_id)
            
            # Color box
            cv2.rectangle(
                annotated,
                (start_x, y - 10),
                (start_x + 20, y + 5),
                color,
                -1
            )
            
            # Track ID text
            cv2.putText(
                annotated,
                f"ID {track_id}",
                (start_x + 30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.COLORS['white'],
                1
            )
        
        return annotated
    
    def draw_full_tracking_visualization(
        self,
        frame: np.ndarray,
        trajectory_manager: TrajectoryManager,
        track_ids: np.ndarray,
        track_boxes: np.ndarray,
        current_frame: int,
        show_trajectories: bool = True,
        show_velocities: bool = True,
        show_stats: bool = True,
        show_legend: bool = True
    ) -> np.ndarray:
        """
        Draw complete tracking visualization with all features.
        
        Args:
            frame: Input frame
            trajectory_manager: Trajectory manager
            track_ids: Array of track IDs
            track_boxes: Array of bboxes [N, 4]
            current_frame: Current frame index
            show_trajectories: Whether to show trajectory paths
            show_velocities: Whether to show velocity arrows
            show_stats: Whether to show stats panel
            show_legend: Whether to show track legend
        
        Returns:
            Fully annotated frame
        """
        annotated = frame.copy()
        
        # Draw trajectories first (background layer)
        if show_trajectories:
            for track_id in trajectory_manager.tracks.keys():
                trajectory = trajectory_manager.get_trajectory(track_id)
                if trajectory is not None and len(trajectory) > 1:
                    annotated = self.draw_trajectory(annotated, trajectory, track_id)
        
        # Draw bounding boxes and labels for active tracks
        for i, track_id in enumerate(track_ids):
            bbox = track_boxes[i]
            
            # Get track data
            velocity = trajectory_manager.get_velocity(track_id, window=5)
            duration = trajectory_manager.get_track_duration(track_id)
            
            # Draw box
            annotated = self.draw_track_box(annotated, track_id, bbox)
            
            # Draw label
            annotated = self.draw_track_label(
                annotated, track_id, bbox, velocity, duration
            )
            
            # Draw velocity arrow
            if show_velocities:
                trajectory = trajectory_manager.get_trajectory(track_id)
                if trajectory is not None and len(trajectory) > 1:
                    centroid = np.array([
                        (bbox[0] + bbox[2]) / 2,
                        (bbox[1] + bbox[3]) / 2
                    ])
                    annotated = self.draw_velocity_arrow(
                        annotated, centroid, trajectory, track_id
                    )
        
        # Draw stats panel
        if show_stats:
            annotated = self.draw_stats_panel(
                annotated, trajectory_manager, current_frame, len(track_ids)
            )
        
        # Draw legend
        if show_legend and len(track_ids) > 0:
            annotated = self.draw_track_legend(annotated, track_ids.tolist())
        
        return annotated



