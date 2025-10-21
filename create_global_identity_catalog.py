"""Create global person identity catalog across all videos."""
import json
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict


def load_all_embeddings_with_metadata():
    """Load all embeddings and metadata from all videos."""
    print("Loading embeddings from all videos...")
    
    gallery_base = Path("data/person_gallery")
    all_tracks = []
    
    # Get all video directories
    video_dirs = sorted([d for d in gallery_base.iterdir() if d.is_dir() and d.name.isdigit()])
    
    for video_dir in video_dirs:
        video_id = video_dir.name
        print(f"\n  Video {video_id}:")
        
        # Get all track directories
        track_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('track_')])
        
        for track_dir in track_dirs:
            track_id = int(track_dir.name.split('_')[1])
            
            # Load embedding
            embedding_path = track_dir / "embedding.npy"
            if not embedding_path.exists():
                continue
            
            embedding = np.load(embedding_path)
            
            # Load metadata
            metadata_path = track_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            
            # Create track entry
            track_entry = {
                'video_id': video_id,
                'track_id': track_id,
                'embedding': embedding,
                'metadata': metadata,
                'unique_key': f"v{video_id}_t{track_id:03d}"
            }
            
            all_tracks.append(track_entry)
            print(f"    Track {track_id}: {metadata.get('total_frames', '?')} frames")
    
    print(f"\n✓ Loaded {len(all_tracks)} tracks from {len(video_dirs)} videos")
    return all_tracks


def compute_similarity_matrix(tracks):
    """Compute pairwise cosine similarity matrix."""
    print("\nComputing similarity matrix...")
    
    n = len(tracks)
    embeddings = np.array([t['embedding'] for t in tracks])
    
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    print(f"  Matrix shape: {similarity_matrix.shape}")
    print(f"  Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    
    return similarity_matrix


def cluster_tracks(tracks, similarity_matrix, threshold=0.60):
    """Cluster tracks by similarity using hierarchical clustering."""
    print(f"\nClustering tracks with threshold {threshold}...")
    
    n = len(tracks)
    
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Hierarchical clustering
    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed, method='average')
    
    # Get cluster labels
    labels = fcluster(linkage_matrix, 1 - threshold, criterion='distance')
    
    # Create cluster mapping
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    
    print(f"  Found {len(clusters)} unique persons")
    
    # Sort clusters by size (number of appearances)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    return sorted_clusters, labels


def create_global_catalog(tracks, sorted_clusters, similarity_matrix):
    """Create the global person identity catalog."""
    print("\nCreating global identity catalog...")
    
    catalog = {}
    
    for global_id, (cluster_label, track_indices) in enumerate(sorted_clusters, start=1):
        person_key = f"global_person_{global_id:03d}"
        
        appearances = []
        videos_set = set()
        
        for idx in track_indices:
            track = tracks[idx]
            metadata = track['metadata']
            
            appearance = {
                'clip_id': track['video_id'],
                'track_id': track['track_id'],
                'frame_range': [0, metadata.get('selected_frame_idx', 0)],
                'total_frames': metadata.get('total_frames', 0),
                'avg_confidence': metadata.get('avg_confidence', 0),
                'quality_score': metadata.get('quality_score', 0),
                'selected_frame_idx': metadata.get('selected_frame_idx', 0)
            }
            
            appearances.append(appearance)
            videos_set.add(track['video_id'])
        
        # Calculate intra-cluster similarities
        if len(track_indices) > 1:
            intra_sims = []
            for i, idx_i in enumerate(track_indices):
                for idx_j in track_indices[i+1:]:
                    intra_sims.append(similarity_matrix[idx_i][idx_j])
            avg_intra_sim = float(np.mean(intra_sims))
            min_intra_sim = float(np.min(intra_sims))
            max_intra_sim = float(np.max(intra_sims))
        else:
            avg_intra_sim = 1.0
            min_intra_sim = 1.0
            max_intra_sim = 1.0
        
        # Sort appearances by video and track
        appearances.sort(key=lambda x: (x['clip_id'], x['track_id']))
        
        catalog[person_key] = {
            'global_id': global_id,
            'appearances': appearances,
            'total_appearances': len(appearances),
            'videos': sorted(list(videos_set)),
            'total_frames': sum(a['total_frames'] for a in appearances),
            'avg_intra_cluster_similarity': round(avg_intra_sim, 4),
            'min_intra_cluster_similarity': round(min_intra_sim, 4),
            'max_intra_cluster_similarity': round(max_intra_sim, 4)
        }
    
    print(f"  Created {len(catalog)} global person identities")
    return catalog


def print_catalog_summary(catalog):
    """Print summary of the catalog."""
    print("\n" + "="*80)
    print("GLOBAL PERSON IDENTITY CATALOG SUMMARY")
    print("="*80)
    
    # Overall statistics
    total_appearances = sum(p['total_appearances'] for p in catalog.values())
    multi_video_persons = sum(1 for p in catalog.values() if len(p['videos']) > 1)
    
    print(f"\nTotal unique persons: {len(catalog)}")
    print(f"Total appearances: {total_appearances}")
    print(f"Persons appearing in multiple videos: {multi_video_persons}")
    
    # Per-person breakdown
    print(f"\n{'='*80}")
    print("DETAILED BREAKDOWN:")
    print(f"{'='*80}\n")
    
    for person_key, person_data in catalog.items():
        print(f"{person_key} (Global ID: {person_data['global_id']})")
        print(f"  Appearances: {person_data['total_appearances']}")
        print(f"  Videos: {', '.join(person_data['videos'])}")
        print(f"  Total frames: {person_data['total_frames']}")
        
        if person_data['total_appearances'] > 1:
            print(f"  Intra-cluster similarity: {person_data['avg_intra_cluster_similarity']:.4f} "
                  f"[{person_data['min_intra_cluster_similarity']:.4f} - "
                  f"{person_data['max_intra_cluster_similarity']:.4f}]")
        
        print(f"  Tracks:")
        for app in person_data['appearances']:
            print(f"    • Video {app['clip_id']}, Track {app['track_id']:03d}: "
                  f"{app['total_frames']} frames (confidence: {app['avg_confidence']:.3f})")
        print()


def save_catalog(catalog, output_path="data/person_gallery/global_identity_catalog.json"):
    """Save catalog to JSON file."""
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"✓ Catalog saved to: {output_path}")
    return output_path


def create_track_mapping(tracks, labels):
    """Create a mapping from original tracks to global IDs."""
    mapping = {}
    
    for idx, track in enumerate(tracks):
        global_id = int(labels[idx])
        key = f"video_{track['video_id']}_track_{track['track_id']:03d}"
        
        mapping[key] = {
            'video_id': track['video_id'],
            'track_id': track['track_id'],
            'global_id': global_id
        }
    
    return mapping


def save_track_mapping(mapping, output_path="data/person_gallery/track_to_global_id_mapping.json"):
    """Save track to global ID mapping."""
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✓ Track mapping saved to: {output_path}")
    return output_path


def main():
    """Main function."""
    print("="*80)
    print("CREATING GLOBAL PERSON IDENTITY CATALOG")
    print("="*80)
    
    # Configuration
    SIMILARITY_THRESHOLD = 0.60  # Based on our experiments
    
    print(f"\nConfiguration:")
    print(f"  Model: osnet_x1_0 (existing embeddings)")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD}")
    
    # Load all embeddings
    tracks = load_all_embeddings_with_metadata()
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(tracks)
    
    # Cluster tracks
    sorted_clusters, labels = cluster_tracks(tracks, similarity_matrix, SIMILARITY_THRESHOLD)
    
    # Create global catalog
    catalog = create_global_catalog(tracks, sorted_clusters, similarity_matrix)
    
    # Create track mapping
    mapping = create_track_mapping(tracks, labels)
    
    # Print summary
    print_catalog_summary(catalog)
    
    # Save results
    catalog_path = save_catalog(catalog)
    mapping_path = save_track_mapping(mapping)
    
    print("\n" + "="*80)
    print("✅ DONE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  1. {catalog_path}")
    print(f"  2. {mapping_path}")
    print()


if __name__ == "__main__":
    main()

