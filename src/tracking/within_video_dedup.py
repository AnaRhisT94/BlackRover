"""Within-video track deduplication using embedding clustering."""
import numpy as np
from typing import Dict, List, Tuple, Set
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: Array of embeddings [N, embedding_dim]
    
    Returns:
        Similarity matrix [N, N]
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-8)
    
    # Compute cosine similarity
    similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)
    
    return similarity_matrix


def cluster_tracks_by_similarity(
    track_embeddings: Dict[int, np.ndarray],
    similarity_threshold: float = 0.7
) -> Dict[int, int]:
    """
    Cluster tracks within a video based on embedding similarity.
    
    Args:
        track_embeddings: Dictionary mapping track_id -> embedding
        similarity_threshold: Threshold for considering tracks as same person
    
    Returns:
        Dictionary mapping original track_id -> cluster_id
    """
    if len(track_embeddings) == 0:
        return {}
    
    if len(track_embeddings) == 1:
        # Only one track, it's its own cluster
        track_id = list(track_embeddings.keys())[0]
        return {track_id: track_id}
    
    # Extract track IDs and embeddings
    track_ids = list(track_embeddings.keys())
    embeddings = np.array([track_embeddings[tid] for tid in track_ids])
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Convert similarity to distance for clustering
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0
    
    # Convert to condensed distance matrix for scipy
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Cut the dendrogram at the threshold
    distance_threshold = 1 - similarity_threshold
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    # Create mapping from track_id to cluster_id
    # Use the first track_id in each cluster as the cluster representative
    track_to_cluster = {}
    cluster_representatives = {}
    
    for track_id, cluster_label in zip(track_ids, cluster_labels):
        if cluster_label not in cluster_representatives:
            cluster_representatives[cluster_label] = track_id
        track_to_cluster[track_id] = cluster_representatives[cluster_label]
    
    return track_to_cluster


def get_cluster_average_embedding(
    track_embeddings: Dict[int, np.ndarray],
    track_to_cluster: Dict[int, int]
) -> Dict[int, np.ndarray]:
    """
    Compute average embedding for each cluster.
    
    Args:
        track_embeddings: Dictionary mapping track_id -> embedding
        track_to_cluster: Dictionary mapping track_id -> cluster_id
    
    Returns:
        Dictionary mapping cluster_id -> average_embedding
    """
    # Group embeddings by cluster
    cluster_embeddings = {}
    for track_id, cluster_id in track_to_cluster.items():
        if cluster_id not in cluster_embeddings:
            cluster_embeddings[cluster_id] = []
        cluster_embeddings[cluster_id].append(track_embeddings[track_id])
    
    # Compute average embedding for each cluster
    cluster_avg_embeddings = {}
    for cluster_id, embeddings_list in cluster_embeddings.items():
        avg_embedding = np.mean(embeddings_list, axis=0)
        # L2 normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        cluster_avg_embeddings[cluster_id] = avg_embedding
    
    return cluster_avg_embeddings


def deduplicate_video_tracks(
    track_embeddings: Dict[int, np.ndarray],
    similarity_threshold: float = 0.7
) -> Tuple[Dict[int, int], Dict[int, np.ndarray], Dict[int, List[int]]]:
    """
    Deduplicate tracks within a video.
    
    Args:
        track_embeddings: Dictionary mapping track_id -> embedding
        similarity_threshold: Threshold for merging tracks
    
    Returns:
        Tuple of:
        - track_to_cluster: Mapping from original track_id to cluster_id
        - cluster_embeddings: Average embedding for each cluster
        - cluster_members: List of track IDs in each cluster
    """
    # Cluster tracks
    track_to_cluster = cluster_tracks_by_similarity(track_embeddings, similarity_threshold)
    
    # Get average embeddings
    cluster_embeddings = get_cluster_average_embedding(track_embeddings, track_to_cluster)
    
    # Build cluster membership lists
    cluster_members = {}
    for track_id, cluster_id in track_to_cluster.items():
        if cluster_id not in cluster_members:
            cluster_members[cluster_id] = []
        cluster_members[cluster_id].append(track_id)
    
    return track_to_cluster, cluster_embeddings, cluster_members


