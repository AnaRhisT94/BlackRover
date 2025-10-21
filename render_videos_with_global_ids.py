"""Render videos with global person IDs instead of local track IDs."""
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_global_mapping():
    """Load the track to global ID mapping."""
    mapping_path = Path("data/person_gallery/track_to_global_id_mapping.json")
    
    with open(mapping_path) as f:
        mapping = json.load(f)
    
    # Reorganize by video and track for quick lookup
    video_track_map = {}
    for key, data in mapping.items():
        video_id = data['video_id']
        track_id = data['track_id']
        global_id = data['global_id']
        
        if video_id not in video_track_map:
            video_track_map[video_id] = {}
        
        video_track_map[video_id][track_id] = global_id
    
    return video_track_map


def load_person_gallery_data(video_id):
    """Load person gallery data for a video to get track information."""
    summary_path = Path(f"data/person_gallery/{video_id}/summary.json")
    
    if not summary_path.exists():
        return {}
    
    with open(summary_path) as f:
        data = json.load(f)
    
    # Create mapping of frame_idx -> track detections
    # We'll need to reconstruct this from the saved frames
    tracks = {}
    video_dir = Path(f"data/person_gallery/{video_id}")
    
    for track_dir in video_dir.glob("track_*"):
        track_id = int(track_dir.name.split('_')[1])
        metadata_path = track_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            tracks[track_id] = metadata
    
    return tracks


def load_trajectory_data():
    """Load trajectory data if available."""
    # Check if we have saved trajectory/tracking data
    # This would typically be saved during the main processing pipeline
    # For now, we'll work with the person gallery data
    return None


def draw_detection_with_global_id(frame, bbox, global_id, track_id, confidence):
    """Draw bounding box with global ID."""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Color based on global ID (cycle through colors)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    color = colors[(global_id - 1) % len(colors)]
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    label = f"Person {global_id:03d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Background rectangle
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
    
    # Text
    cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    # Show confidence and original track ID (smaller, below box)
    info_label = f"T{track_id} ({confidence:.2f})"
    cv2.putText(frame, info_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, color, 1)
    
    return frame


def process_video_simple(video_id, video_track_map, input_path, output_path):
    """
    Process video with simple re-rendering approach.
    
    Since we don't have frame-by-frame detection data saved, we'll create
    a visualization showing the global IDs on a title card and reference grid.
    """
    print(f"\nProcessing Video {video_id}...")
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"  ✗ Could not open video: {input_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"  ✗ Could not create output video: {output_path}")
        cap.release()
        return False
    
    # Load person gallery images for this video
    gallery_dir = Path(f"data/person_gallery/{video_id}")
    track_images = {}
    track_global_ids = {}
    
    if video_id in video_track_map:
        for track_id, global_id in video_track_map[video_id].items():
            track_dir = gallery_dir / f"track_{track_id:03d}"
            if track_dir.exists():
                # Find the image
                images = list(track_dir.glob("frame_*.jpg"))
                if images:
                    img = cv2.imread(str(images[0]))
                    if img is not None:
                        track_images[track_id] = img
                        track_global_ids[track_id] = global_id
    
    # Create an overlay showing the mapping
    overlay_height = 150
    overlay = np.ones((overlay_height, width, 3), dtype=np.uint8) * 50
    
    # Draw title
    cv2.putText(overlay, f"Video {video_id} - Global Person IDs", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw track mappings
    if track_global_ids:
        y_offset = 60
        x_offset = 10
        
        for track_id, global_id in sorted(track_global_ids.items()):
            text = f"Track {track_id:03d} = Global Person {global_id:03d}"
            
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
            ]
            color = colors[(global_id - 1) % len(colors)]
            
            cv2.putText(overlay, text, (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            y_offset += 25
            if y_offset > overlay_height - 20:
                x_offset += 300
                y_offset = 60
    
    # Process frames
    frame_count = 0
    pbar = tqdm(total=total_frames, desc=f"  Video {video_id}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add overlay at top
        # Make space by shrinking frame
        frame_resized = cv2.resize(frame, (width, height - overlay_height))
        
        # Combine
        output_frame = np.vstack([overlay, frame_resized])
        
        # Write frame
        out.write(output_frame)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"  ✓ Processed {frame_count} frames")
    print(f"  ✓ Saved to: {output_path}")
    
    return True


def create_reference_video(video_id, video_track_map, output_path):
    """Create a reference video showing the person gallery with global IDs."""
    print(f"\nCreating reference video for Video {video_id}...")
    
    gallery_dir = Path(f"data/person_gallery/{video_id}")
    
    # Load all track images
    track_data = []
    
    if video_id in video_track_map:
        for track_id, global_id in sorted(video_track_map[video_id].items()):
            track_dir = gallery_dir / f"track_{track_id:03d}"
            if track_dir.exists():
                images = list(track_dir.glob("frame_*.jpg"))
                metadata_path = track_dir / "metadata.json"
                
                if images and metadata_path.exists():
                    img = cv2.imread(str(images[0]))
                    
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    
                    if img is not None:
                        track_data.append({
                            'track_id': track_id,
                            'global_id': global_id,
                            'image': img,
                            'metadata': metadata
                        })
    
    if not track_data:
        print(f"  ✗ No track data found")
        return False
    
    # Create grid
    n_tracks = len(track_data)
    cols = min(4, n_tracks)
    rows = (n_tracks + cols - 1) // cols
    
    # Resize all images to standard size
    img_w, img_h = 200, 400
    padding = 20
    
    grid_w = cols * (img_w + padding) + padding
    grid_h = rows * (img_h + 150 + padding) + padding + 100  # Extra for title
    
    # Create reference frame
    frame = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # Title
    title = f"Video {video_id} - Global Person IDs"
    cv2.rectangle(frame, (0, 0), (grid_w, 80), (50, 50, 50), -1)
    cv2.putText(frame, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Place images
    for idx, data in enumerate(track_data):
        row = idx // cols
        col = idx % cols
        
        x = padding + col * (img_w + padding)
        y = 100 + padding + row * (img_h + 150 + padding)
        
        # Resize image
        img_resized = cv2.resize(data['image'], (img_w, img_h))
        
        # Place image
        frame[y:y+img_h, x:x+img_w] = img_resized
        
        # Draw border with global ID color
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        color = colors[(data['global_id'] - 1) % len(colors)]
        cv2.rectangle(frame, (x-2, y-2), (x+img_w+2, y+img_h+2), color, 4)
        
        # Label background
        label_y = y + img_h
        cv2.rectangle(frame, (x, label_y), (x+img_w, label_y+150), (240, 240, 240), -1)
        
        # Labels
        labels = [
            f"Global: {data['global_id']:03d}",
            f"Track: {data['track_id']:03d}",
            f"Frames: {data['metadata'].get('total_frames', 0)}",
            f"Conf: {data['metadata'].get('avg_confidence', 0):.3f}"
        ]
        
        for i, label in enumerate(labels):
            cv2.putText(frame, label, (x+5, label_y+25+i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save as video (static frame for 5 seconds)
    fps = 30
    duration = 5
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_w, grid_h))
    
    if not out.isOpened():
        print(f"  ✗ Could not create video")
        return False
    
    # Write frames
    for _ in range(total_frames):
        out.write(frame)
    
    out.release()
    
    print(f"  ✓ Saved to: {output_path}")
    return True


def main():
    """Main function."""
    print("="*80)
    print("RENDERING VIDEOS WITH GLOBAL PERSON IDs")
    print("="*80)
    
    # Load mapping
    print("\nLoading global ID mapping...")
    video_track_map = load_global_mapping()
    
    print(f"Loaded mappings for {len(video_track_map)} videos")
    for vid, tracks in video_track_map.items():
        print(f"  Video {vid}: {len(tracks)} tracks")
    
    # Setup paths
    input_dir = Path("input")
    output_dir = Path("data/output_videos_global_ids")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each video
    video_files = {
        '1': input_dir / "1.mp4",
        '2': input_dir / "2.mp4",
        '3': input_dir / "3.mp4",
        '4': input_dir / "4.mp4"
    }
    
    success_count = 0
    
    for video_id, input_path in sorted(video_files.items()):
        if not input_path.exists():
            print(f"\n✗ Video {video_id} not found: {input_path}")
            continue
        
        # Create reference video (gallery with global IDs)
        ref_output = output_dir / f"video_{video_id}_reference.mp4"
        if create_reference_video(video_id, video_track_map, ref_output):
            success_count += 1
        
        # Process main video with overlay
        output_path = output_dir / f"video_{video_id}_with_global_ids.mp4"
        if process_video_simple(video_id, video_track_map, input_path, output_path):
            success_count += 1
    
    print("\n" + "="*80)
    print(f"✅ DONE! Processed {success_count} outputs")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.glob("*.mp4")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

