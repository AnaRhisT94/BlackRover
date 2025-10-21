"""DeepSORT tracker for multi-object tracking with re-identification."""
import numpy as np
from typing import List, Dict, Tuple, Optional
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv


class PersonTracker:
    """DeepSORT-based person tracker with re-identification."""
    
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        nms_max_overlap: float = 1.0,
        max_cosine_distance: float = 0.3,
        nn_budget: Optional[int] = 100,
        embedder: str = "torchreid",
        embedder_model_name: str = "osnet_x1_0",
        embedder_gpu: bool = True
    ):
        """
        Initialize DeepSORT tracker.
        
        Args:
            max_age: Maximum number of frames to keep alive a track without matching detections
            n_init: Number of consecutive detections before track is confirmed
            nms_max_overlap: Non-maximum suppression threshold
            max_cosine_distance: Maximum cosine distance for appearance matching
            nn_budget: Maximum size of appearance descriptors gallery
            embedder: Embedding model backend ('torchreid' or 'clip')
            embedder_model_name: Name of the embedder model
            embedder_gpu: Whether to use GPU for embedder
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder=embedder,
            embedder_model_name=embedder_model_name,
            embedder_gpu=embedder_gpu
        )
        
        # Store track history
        self.track_history: Dict[int, List[np.ndarray]] = {}
        
        # Store current tracks for embedding access
        self.current_tracks = []
        
    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Supervision Detections object with person bboxes
            frame: Current frame (for appearance embeddings)
        
        Returns:
            Tuple of:
                - track_ids: Array of track IDs [N]
                - track_boxes: Array of bboxes [N, 4] (x1, y1, x2, y2)
                - track_confidences: Array of confidences [N]
        """
        if len(detections) == 0:
            # Update with empty detections to age out tracks
            self.tracker.update_tracks([], frame=frame)
            return np.array([]), np.array([]), np.array([])
        
        # Convert supervision detections to DeepSORT format
        # DeepSORT expects: [([left, top, w, h], confidence, class), ...]
        raw_detections = []
        detection_confidences = []
        
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            w = x2 - x1
            h = y2 - y1
            conf = detections.confidence[i] if detections.confidence is not None else 1.0
            
            # Format as tuple: ([bbox], confidence, class)
            bbox = [float(x1), float(y1), float(w), float(h)]
            raw_detections.append((bbox, float(conf), 'person'))
            detection_confidences.append(float(conf))
        
        # Update tracks
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        
        # Store tracks for embedding access
        self.current_tracks = tracks
        
        # Create mapping from detection index to track
        # This is needed to properly assign confidences
        det_to_track_conf = {}
        for i, track in enumerate(tracks):
            if hasattr(track, 'det_conf') and track.det_conf is not None:
                det_to_track_conf[track.track_id] = track.det_conf
        
        # Extract confirmed tracks
        track_ids = []
        track_boxes = []
        track_confs = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            
            # Get the confidence for this track
            conf = det_to_track_conf.get(track_id, 1.0)
            
            track_ids.append(track_id)
            track_boxes.append(ltrb)
            track_confs.append(conf)
            
            # Store track history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            # Store centroid
            centroid = np.array([
                (ltrb[0] + ltrb[2]) / 2,
                (ltrb[1] + ltrb[3]) / 2
            ])
            self.track_history[track_id].append(centroid)
            
            # Keep only recent history (last 60 frames)
            if len(self.track_history[track_id]) > 60:
                self.track_history[track_id] = self.track_history[track_id][-60:]
        
        return (
            np.array(track_ids, dtype=int),
            np.array(track_boxes, dtype=float),
            np.array(track_confs, dtype=float)
        )
    
    def get_trajectory(self, track_id: int) -> Optional[np.ndarray]:
        """
        Get trajectory history for a track.
        
        Args:
            track_id: Track ID
        
        Returns:
            Array of centroids [T, 2] or None if track not found
        """
        if track_id in self.track_history:
            return np.array(self.track_history[track_id])
        return None
    
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
        
        recent = trajectory[-window:]
        if len(recent) < 2:
            return None
        
        # Calculate displacements
        displacements = np.linalg.norm(np.diff(recent, axis=0), axis=1)
        return np.mean(displacements)
    
    def get_embeddings(self, track_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings for specified track IDs.
        
        Args:
            track_ids: Array of track IDs to get embeddings for
        
        Returns:
            Tuple of:
                - embeddings: Array of embeddings [N, embedding_dim] or empty array
                - valid_mask: Boolean array indicating which track IDs have embeddings [N]
        """
        if len(track_ids) == 0 or len(self.current_tracks) == 0:
            return np.array([]), np.array([], dtype=bool)
        
        # Build mapping from track_id to track object
        track_map = {}
        for track in self.current_tracks:
            # Include both confirmed and tentative tracks that have embeddings
            if track.is_confirmed() or track.is_tentative():
                # Convert track ID to int for consistency (deep_sort uses strings internally)
                track_map[int(track.track_id)] = track
        
        valid_embeddings = []
        valid_indices = []
        
        for i, track_id in enumerate(track_ids):
            # Ensure track_id is int
            track_id_int = int(track_id)
            if track_id_int in track_map:
                track = track_map[track_id_int]
                # Get the most recent feature/embedding from the track
                # features is a list, get the last one
                if hasattr(track, 'features') and len(track.features) > 0:
                    # Get the most recent embedding (last in list)
                    embedding = track.features[-1]
                    valid_embeddings.append(embedding)
                    valid_indices.append(i)
        
        # Create valid_mask
        valid_mask = np.zeros(len(track_ids), dtype=bool)
        if valid_indices:
            valid_mask[valid_indices] = True
        
        if valid_embeddings:
            return np.array(valid_embeddings), valid_mask
        else:
            return np.array([]), valid_mask
    
    def reset(self):
        """Reset tracker and clear all tracks."""
        self.tracker = DeepSort(
            max_age=self.tracker.max_age,
            n_init=self.tracker.n_init,
            nms_max_overlap=self.tracker.nms_max_overlap,
            max_cosine_distance=self.tracker.max_cosine_distance,
            nn_budget=self.tracker.nn_budget
        )
        self.track_history.clear()
    
    def __repr__(self):
        active_tracks = len([t for t in self.tracker.tracks if t.is_confirmed()])
        return f"PersonTracker(active_tracks={active_tracks}, total_history={len(self.track_history)})"

