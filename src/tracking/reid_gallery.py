"""Cross-video Re-Identification Gallery for person matching."""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class GalleryEntry:
    """Represents a person in the gallery."""
    global_id: int
    embedding: np.ndarray  # L2-normalized embedding
    source_video: str
    source_track_id: int
    confidence: float
    match_count: int = 1  # Number of times this embedding has been updated/matched
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'global_id': self.global_id,
            'embedding': self.embedding.tolist(),
            'source_video': self.source_video,
            'source_track_id': int(self.source_track_id),
            'confidence': float(self.confidence),
            'match_count': self.match_count
        }


class CrossVideoReIDGallery:
    """
    Manages a gallery of person embeddings across multiple videos for re-identification.
    
    Features:
    - Progressive gallery building across videos
    - Embedding averaging when matches are found
    - Global ID assignment and tracking
    """
    
    def __init__(self, similarity_threshold: float = 0.7, output_dir: Optional[str] = None):
        """
        Initialize the ReID gallery.
        
        Args:
            similarity_threshold: Cosine similarity threshold for matching (default: 0.7)
            output_dir: Directory to save gallery data (optional)
        """
        self.similarity_threshold = similarity_threshold
        self.output_dir = Path(output_dir) if output_dir else None
        self.gallery: Dict[int, GalleryEntry] = {}
        self.next_global_id = 1
        self.current_video_name = None
        
        # Statistics
        self.stats = {
            'total_persons': 0,
            'total_matches': 0,
            'total_new_entries': 0
        }
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Embeddings should already be normalized, but ensure it
        emb1_norm = self._normalize_embedding(emb1)
        emb2_norm = self._normalize_embedding(emb2)
        return float(np.dot(emb1_norm, emb2_norm))
    
    def set_current_video(self, video_name: str):
        """Set the current video being processed."""
        self.current_video_name = video_name
    
    def match_or_add(
        self,
        embedding: np.ndarray,
        track_id: int,
        confidence: float,
        video_name: Optional[str] = None
    ) -> Tuple[int, bool]:
        """
        Match an embedding against the gallery or add as new person.
        
        Args:
            embedding: Feature embedding from OSNet
            track_id: Local track ID in current video
            confidence: Detection confidence
            video_name: Name of source video (uses current_video_name if None)
        
        Returns:
            Tuple of (global_id, is_new)
            - global_id: Global person ID assigned
            - is_new: True if new person added, False if matched existing
        """
        video_name = video_name or self.current_video_name
        if video_name is None:
            raise ValueError("Video name must be set via set_current_video() or passed as argument")
        
        # Normalize the input embedding
        embedding = self._normalize_embedding(embedding)
        
        # If gallery is empty, add as first person
        if not self.gallery:
            global_id = self.next_global_id
            self.next_global_id += 1
            
            self.gallery[global_id] = GalleryEntry(
                global_id=global_id,
                embedding=embedding,
                source_video=video_name,
                source_track_id=track_id,
                confidence=confidence
            )
            
            self.stats['total_persons'] += 1
            self.stats['total_new_entries'] += 1
            return global_id, True
        
        # Find best match in gallery
        best_match_id = None
        best_similarity = -1.0
        
        for global_id, entry in self.gallery.items():
            similarity = self._compute_similarity(embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = global_id
        
        # Check if best match exceeds threshold
        if best_similarity >= self.similarity_threshold:
            # Match found! Update the gallery embedding by averaging
            entry = self.gallery[best_match_id]
            
            # Weighted average: current embedding gets weight based on match count
            # This gives more weight to the established gallery embedding
            weight_old = entry.match_count
            weight_new = 1
            total_weight = weight_old + weight_new
            
            updated_embedding = (entry.embedding * weight_old + embedding * weight_new) / total_weight
            entry.embedding = self._normalize_embedding(updated_embedding)
            entry.match_count += 1
            
            self.stats['total_matches'] += 1
            return best_match_id, False
        else:
            # No match found, add as new person
            global_id = self.next_global_id
            self.next_global_id += 1
            
            self.gallery[global_id] = GalleryEntry(
                global_id=global_id,
                embedding=embedding,
                source_video=video_name,
                source_track_id=track_id,
                confidence=confidence
            )
            
            self.stats['total_persons'] += 1
            self.stats['total_new_entries'] += 1
            return global_id, True
    
    def match_batch(
        self,
        embeddings: np.ndarray,
        track_ids: np.ndarray,
        confidences: np.ndarray,
        video_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match a batch of embeddings against the gallery.
        
        Args:
            embeddings: Array of embeddings, shape (N, embedding_dim)
            track_ids: Array of local track IDs, shape (N,)
            confidences: Array of confidences, shape (N,)
            video_name: Name of source video
        
        Returns:
            Tuple of (global_ids, is_new_mask)
            - global_ids: Array of assigned global IDs, shape (N,)
            - is_new_mask: Boolean array indicating new persons, shape (N,)
        """
        global_ids = []
        is_new_list = []
        
        for emb, track_id, conf in zip(embeddings, track_ids, confidences):
            global_id, is_new = self.match_or_add(emb, track_id, conf, video_name)
            global_ids.append(global_id)
            is_new_list.append(is_new)
        
        return np.array(global_ids), np.array(is_new_list)
    
    def get_statistics(self) -> dict:
        """Get gallery statistics."""
        return {
            'total_persons': self.stats['total_persons'],
            'total_matches': self.stats['total_matches'],
            'total_new_entries': self.stats['total_new_entries'],
            'gallery_size': len(self.gallery),
            'avg_match_count': np.mean([e.match_count for e in self.gallery.values()]) if self.gallery else 0
        }
    
    def save_gallery(self, filepath: Optional[str] = None):
        """Save gallery to JSON file."""
        if filepath is None:
            if self.output_dir is None:
                raise ValueError("No output directory or filepath specified")
            filepath = self.output_dir / "reid_gallery.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'similarity_threshold': self.similarity_threshold,
            'next_global_id': self.next_global_id,
            'statistics': self.get_statistics(),
            'gallery': [entry.to_dict() for entry in self.gallery.values()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load_gallery(self, filepath: str):
        """Load gallery from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.similarity_threshold = data['similarity_threshold']
        self.next_global_id = data['next_global_id']
        self.stats = data['statistics']
        
        self.gallery = {}
        for entry_dict in data['gallery']:
            entry = GalleryEntry(
                global_id=entry_dict['global_id'],
                embedding=np.array(entry_dict['embedding']),
                source_video=entry_dict['source_video'],
                source_track_id=entry_dict['source_track_id'],
                confidence=entry_dict['confidence'],
                match_count=entry_dict.get('match_count', 1)
            )
            self.gallery[entry.global_id] = entry
    
    def get_global_id_mapping(self, video_name: str) -> Dict[int, int]:
        """
        Get mapping from local track IDs to global IDs for a specific video.
        
        Returns:
            Dictionary mapping local_track_id -> global_id
        """
        mapping = {}
        for entry in self.gallery.values():
            if entry.source_video == video_name:
                mapping[entry.source_track_id] = entry.global_id
        return mapping
    
    def __len__(self):
        """Return number of unique persons in gallery."""
        return len(self.gallery)
    
    def __repr__(self):
        return (f"CrossVideoReIDGallery(persons={len(self.gallery)}, "
                f"threshold={self.similarity_threshold}, "
                f"matches={self.stats['total_matches']})")


