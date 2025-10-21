"""Person gallery builder for cross-video re-identification."""
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class FrameCandidate:
    """Candidate frame for person gallery."""
    frame_idx: int
    confidence: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    frame_crop: np.ndarray  # Cropped person image
    width: int
    height: int
    isolation_score: float  # Distance to nearest other person (0=overlap, 1=isolated)
    quality_score: float  # Overall quality metric
    has_overlap: bool  # Whether this crop overlaps with other people
    num_keypoints: int  # Number of detected keypoints above threshold


@dataclass
class TrackMetadata:
    """Metadata for a tracked person."""
    track_id: int
    video_name: str
    total_frames: int
    avg_confidence: float
    selected_frame_idx: int
    selected_confidence: float
    width: int
    height: int
    isolation_score: float
    quality_score: float
    num_keypoints: int  # Number of detected keypoints in selected frame
    bbox_average: List[float]  # [width, height] averaged over all frames


class PersonGalleryBuilder:
    """
    Builds a gallery of person images for cross-video re-identification.
    
    For each tracked person, selects the single best frame based on quality scoring
    (confidence, size, isolation), filters out multi-person crops, and creates
    a composite grid image showing all selected crops.
    """
    
    def __init__(
        self,
        output_dir: str,
        video_name: str,
        min_confidence: float = 0.7,
        max_iou_threshold: float = 0.15,
        min_isolation_distance: float = 50.0
    ):
        """
        Initialize gallery builder.
        
        Args:
            output_dir: Base output directory (e.g., "data/person_gallery")
            video_name: Name of the video being processed
            min_confidence: Minimum confidence threshold for frame selection
            max_iou_threshold: Maximum IoU with other detections (to ensure single person)
            min_isolation_distance: Minimum pixel distance to nearest person for best score
        """
        self.output_dir = Path(output_dir)
        self.video_name = video_name
        self.min_confidence = min_confidence
        self.max_iou_threshold = max_iou_threshold
        self.min_isolation_distance = min_isolation_distance
        
        # Video-specific output directory
        self.video_dir = self.output_dir / video_name
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # Track data: track_id -> best FrameCandidate
        self.track_best_frame: Dict[int, FrameCandidate] = {}
        
        # Track statistics (all frames seen)
        self.track_stats: Dict[int, Dict] = {}
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _compute_min_distance(self, bbox: np.ndarray, other_boxes: List[np.ndarray]) -> float:
        """
        Compute minimum center-to-center distance to other boxes.
        
        Args:
            bbox: Current bounding box [x1, y1, x2, y2]
            other_boxes: List of other bounding boxes
            
        Returns:
            Minimum distance to nearest box (in pixels)
        """
        if not other_boxes:
            return float('inf')
        
        # Center of current box
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        min_dist = float('inf')
        for other_box in other_boxes:
            other_cx = (other_box[0] + other_box[2]) / 2
            other_cy = (other_box[1] + other_box[3]) / 2
            dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _compute_quality_score(
        self,
        confidence: float,
        width: int,
        height: int,
        isolation_score: float,
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Compute overall quality score for a frame.
        
        Scoring factors:
        - Confidence (0-1): 40% weight
        - Size (normalized by frame): 30% weight
        - Isolation: 30% weight
        
        Args:
            confidence: Detection confidence
            width: Crop width
            height: Crop height
            isolation_score: Isolation score (0-1)
            frame_width: Full frame width
            frame_height: Full frame height
            
        Returns:
            Quality score (0-1)
        """
        # Normalize size (0-1 based on frame dimensions)
        size_score = min(1.0, (width * height) / (frame_width * frame_height * 0.5))
        
        # Weighted combination
        quality = (
            0.4 * confidence +
            0.3 * size_score +
            0.3 * isolation_score
        )
        
        return quality
    
    def add_frame(
        self,
        frame_idx: int,
        track_id: int,
        bbox: np.ndarray,
        confidence: float,
        frame: np.ndarray,
        all_boxes: List[np.ndarray],
        keypoints: Optional[np.ndarray] = None,
        keypoint_conf: Optional[np.ndarray] = None,
        keypoint_threshold: float = 0.5
    ):
        """
        Add a frame candidate for a tracked person.
        
        Args:
            frame_idx: Frame index in video
            track_id: Person track ID
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence
            frame: Full frame image
            all_boxes: List of ALL bounding boxes in this frame (for overlap detection)
            keypoints: Keypoint array [17, 2] for this person (optional)
            keypoint_conf: Keypoint confidence array [17] for this person (optional)
            keypoint_threshold: Threshold for counting detected keypoints
        """
        # Skip low confidence detections
        if confidence < self.min_confidence:
            return
        
        # Crop person from frame
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return  # Invalid bbox
        
        frame_crop = frame[y1:y2, x1:x2].copy()
        width = x2 - x1
        height = y2 - y1
        
        # Count detected keypoints
        num_keypoints = 0
        if keypoints is not None and keypoint_conf is not None:
            num_keypoints = int(np.sum(keypoint_conf >= keypoint_threshold))
        
        # Apply size filtering: keep if (width + height >= 250) OR (num_keypoints >= 8)
        if (width + height < 250) and (num_keypoints < 8):
            return  # Skip small crops without sufficient pose information
        
        # Check overlap with other detections
        other_boxes = [b for b in all_boxes if not np.array_equal(b, bbox)]
        max_iou = 0.0
        has_overlap = False
        
        for other_box in other_boxes:
            iou = self._compute_iou(bbox, other_box)
            max_iou = max(max_iou, iou)
            if iou > self.max_iou_threshold:
                has_overlap = True
                break
        
        # Compute isolation score
        min_distance = self._compute_min_distance(bbox, other_boxes)
        isolation_score = min(1.0, min_distance / self.min_isolation_distance)
        
        # Compute quality score
        frame_height, frame_width = frame.shape[:2]
        quality_score = self._compute_quality_score(
            confidence, width, height, isolation_score, frame_width, frame_height
        )
        
        # Initialize track if not exists
        if track_id not in self.track_stats:
            self.track_stats[track_id] = {
                'total_frames': 0,
                'confidences': [],
                'bboxes': []
            }
        
        # Update statistics
        stats = self.track_stats[track_id]
        stats['total_frames'] += 1
        stats['confidences'].append(confidence)
        stats['bboxes'].append(bbox)
        
        # Create candidate
        candidate = FrameCandidate(
            frame_idx=frame_idx,
            confidence=confidence,
            bbox=bbox,
            frame_crop=frame_crop,
            width=width,
            height=height,
            isolation_score=isolation_score,
            quality_score=quality_score,
            has_overlap=has_overlap,
            num_keypoints=num_keypoints
        )
        
        # Keep only if it's the best frame for this track so far
        # Prioritize: no overlap, then highest quality score
        if track_id not in self.track_best_frame:
            self.track_best_frame[track_id] = candidate
        else:
            current_best = self.track_best_frame[track_id]
            
            # Replace if new candidate is better
            # Priority: 1) No overlap, 2) Higher quality score
            replace = False
            
            if has_overlap and not current_best.has_overlap:
                # Current is better (no overlap vs overlap)
                replace = False
            elif not has_overlap and current_best.has_overlap:
                # New is better (no overlap vs overlap)
                replace = True
            else:
                # Both have same overlap status, compare quality
                replace = quality_score > current_best.quality_score
            
            if replace:
                self.track_best_frame[track_id] = candidate
    
    def compute_embedding(
        self,
        frame_crop: np.ndarray,
        embedder
    ) -> np.ndarray:
        """
        Compute embedding for a single frame crop.
        
        Args:
            frame_crop: Cropped person image
            embedder: Embedder object from DeepSort tracker
        
        Returns:
            Normalized embedding vector
        """
        # The embedder expects BGR image (OpenCV format)
        embedding = embedder.predict([frame_crop])[0]  # Returns list, take first
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def create_grid_image(
        self,
        candidates: List[Tuple[int, FrameCandidate]],
        grid_cols: int = 5,
        cell_padding: int = 10,
        label_height: int = 60
    ) -> np.ndarray:
        """
        Create a grid image showing all selected person crops with labels.
        
        Args:
            candidates: List of (track_id, candidate) tuples
            grid_cols: Number of columns in grid
            cell_padding: Padding between cells in pixels
            label_height: Height reserved for label text
            
        Returns:
            Grid image as numpy array
        """
        if not candidates:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Sort by track_id
        candidates = sorted(candidates, key=lambda x: x[0])
        
        # Determine grid dimensions
        n_tracks = len(candidates)
        grid_rows = (n_tracks + grid_cols - 1) // grid_cols
        
        # Find max dimensions for uniform cell size
        max_width = max(cand.width for _, cand in candidates)
        max_height = max(cand.height for _, cand in candidates)
        
        # Cell dimensions (including label space)
        cell_width = max_width + 2 * cell_padding
        cell_height = max_height + label_height + 2 * cell_padding
        
        # Create grid canvas
        grid_width = grid_cols * cell_width
        grid_height = grid_rows * cell_height
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Place each crop in the grid
        for idx, (track_id, candidate) in enumerate(candidates):
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Calculate position
            y_start = row * cell_height + cell_padding
            x_start = col * cell_width + cell_padding
            
            # Center the crop in the cell
            x_offset = (max_width - candidate.width) // 2
            y_offset = (max_height - candidate.height) // 2
            
            crop_y = y_start + y_offset
            crop_x = x_start + x_offset
            
            # Place the crop
            grid[crop_y:crop_y + candidate.height, crop_x:crop_x + candidate.width] = candidate.frame_crop
            
            # Draw border around crop
            cv2.rectangle(
                grid,
                (crop_x - 2, crop_y - 2),
                (crop_x + candidate.width + 2, crop_y + candidate.height + 2),
                (0, 255, 0) if not candidate.has_overlap else (0, 0, 255),
                2
            )
            
            # Add labels below the crop
            label_y = y_start + max_height + y_offset + 15
            label_x = x_start
            
            # Track ID
            cv2.putText(
                grid,
                f"Track {track_id}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            
            # Size (W×H) and Keypoints
            cv2.putText(
                grid,
                f"{candidate.width}x{candidate.height} | KP:{candidate.num_keypoints}",
                (label_x, label_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )
            
            # Confidence and Quality
            cv2.putText(
                grid,
                f"Conf:{candidate.confidence:.2f} Q:{candidate.quality_score:.2f}",
                (label_x, label_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1
            )
        
        return grid
    
    def save_gallery(self, embedder=None, filter_overlap: bool = True):
        """
        Save the person gallery to disk.
        
        Saves:
        - Individual best frame crop per track
        - Grid image showing all crops with labels
        - Embeddings (if embedder provided)
        - Comprehensive metadata JSON
        
        Args:
            embedder: Optional embedder object from tracker for computing embeddings.
                     If None, embeddings won't be computed.
            filter_overlap: If True, skip tracks where best frame has overlap with other people
        
        Returns:
            Dictionary with statistics about saved tracks
        """
        saved_count = 0
        skipped_overlap = 0
        
        # Collect candidates for grid and saving
        candidates_to_save = []
        
        for track_id, candidate in self.track_best_frame.items():
            # Filter out overlapping crops if requested
            if filter_overlap and candidate.has_overlap:
                skipped_overlap += 1
                continue
            
            candidates_to_save.append((track_id, candidate))
        
        # Create grid image
        if candidates_to_save:
            grid_image = self.create_grid_image(candidates_to_save)
            grid_path = self.video_dir / "gallery_grid.jpg"
            cv2.imwrite(str(grid_path), grid_image)
        
        # Save individual crops and metadata
        all_metadata = []
        
        for track_id, candidate in candidates_to_save:
            # Create track directory
            track_dir = self.video_dir / f"track_{track_id:03d}"
            track_dir.mkdir(exist_ok=True)
            
            # Save the best frame crop
            frame_path = track_dir / f"frame_{candidate.frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), candidate.frame_crop)
            
            # Compute and save embedding if embedder provided
            if embedder is not None:
                try:
                    embedding = self.compute_embedding(candidate.frame_crop, embedder)
                    embedding_path = track_dir / "embedding.npy"
                    np.save(embedding_path, embedding)
                except Exception as e:
                    print(f"Warning: Failed to compute embedding for track {track_id}: {e}")
            
            # Compute track statistics
            stats = self.track_stats[track_id]
            bboxes = np.array(stats['bboxes'])
            widths = bboxes[:, 2] - bboxes[:, 0]
            heights = bboxes[:, 3] - bboxes[:, 1]
            
            # Create metadata
            metadata = TrackMetadata(
                track_id=int(track_id),
                video_name=self.video_name,
                total_frames=int(stats['total_frames']),
                avg_confidence=float(np.mean(stats['confidences'])),
                selected_frame_idx=int(candidate.frame_idx),
                selected_confidence=float(candidate.confidence),
                width=int(candidate.width),
                height=int(candidate.height),
                isolation_score=float(candidate.isolation_score),
                quality_score=float(candidate.quality_score),
                num_keypoints=int(candidate.num_keypoints),
                bbox_average=[float(np.mean(widths)), float(np.mean(heights))]
            )
            
            # Save track metadata
            metadata_path = track_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            all_metadata.append(asdict(metadata))
            saved_count += 1
        
        # Save video-level summary with all metadata
        summary = {
            'video_name': self.video_name,
            'total_tracks': len(self.track_best_frame),
            'saved_tracks': saved_count,
            'skipped_overlap': skipped_overlap,
            'min_confidence': self.min_confidence,
            'max_iou_threshold': self.max_iou_threshold,
            'tracks_metadata': all_metadata
        }
        
        summary_path = self.video_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save a CSV for easy analysis
        if all_metadata:
            import csv
            csv_path = self.video_dir / "tracks_summary.csv"
            with open(csv_path, 'w', newline='') as f:
                fieldnames = ['track_id', 'frame_idx', 'confidence', 'width', 'height', 
                             'num_keypoints', 'isolation_score', 'quality_score', 
                             'total_frames', 'avg_confidence']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for meta in all_metadata:
                    writer.writerow({
                        'track_id': meta['track_id'],
                        'frame_idx': meta['selected_frame_idx'],
                        'confidence': meta['selected_confidence'],
                        'width': meta['width'],
                        'height': meta['height'],
                        'num_keypoints': meta['num_keypoints'],
                        'isolation_score': meta['isolation_score'],
                        'quality_score': meta['quality_score'],
                        'total_frames': meta['total_frames'],
                        'avg_confidence': meta['avg_confidence']
                    })
        
        return summary
    
    def get_statistics(self) -> Dict:
        """Get current statistics about collected frames."""
        tracks_with_best = len(self.track_best_frame)
        tracks_without_overlap = sum(
            1 for candidate in self.track_best_frame.values()
            if not candidate.has_overlap
        )
        
        return {
            'total_tracks': len(self.track_stats),
            'tracks_with_best_frame': tracks_with_best,
            'tracks_without_overlap': tracks_without_overlap,
            'overlap_percentage': 
                (tracks_with_best - tracks_without_overlap) / tracks_with_best * 100 
                if tracks_with_best > 0 else 0
        }

