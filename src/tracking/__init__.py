"""Tracking module for multi-object tracking with DeepSORT."""
from .deepsort_tracker import PersonTracker
from .trajectory_manager import TrajectoryManager, TrackData

__all__ = ['PersonTracker', 'TrajectoryManager', 'TrackData']

