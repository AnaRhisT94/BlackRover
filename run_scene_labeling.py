#!/usr/bin/env python3
"""
Scene Labeling: Run theft and gun detection on all videos
Output: dataset-level JSON with crime labels and justifications
"""

import json
from pathlib import Path
from detect_theft import detect_hand_theft
from test_gun_detection import test_gun_detection


def load_global_identity_catalog(catalog_path="data/person_gallery/global_identity_catalog.json"):
    """Load the global identity catalog to map track IDs to global person IDs."""
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    
    # Create a mapping: clip_id -> track_id -> global_id
    clip_track_to_global = {}
    for person_key, person_data in catalog.items():
        global_id = person_data['global_id']
        for appearance in person_data['appearances']:
            clip_id = str(appearance['clip_id'])
            track_id = appearance['track_id']
            
            if clip_id not in clip_track_to_global:
                clip_track_to_global[clip_id] = {}
            clip_track_to_global[clip_id][track_id] = global_id
    
    return clip_track_to_global


def format_justification(theft_events, gun_events, clip_id, track_to_global, fps):
    """Format a human-readable justification with timestamps and global IDs."""
    justifications = []
    
    # Process theft events
    for event in theft_events:
        track_id = event['track_id']
        global_id = track_to_global.get(clip_id, {}).get(track_id, f"unknown_track_{track_id}")
        theft_time = event['theft_frame'] / fps
        wrist = event['wrist']
        
        # Format global_id properly - may be int or string
        if isinstance(global_id, int):
            global_id_str = f"global_person_{global_id:03d}"
        else:
            global_id_str = str(global_id)
        
        justifications.append(
            f"Theft detected at {theft_time:.1f}s by {global_id_str} ({wrist} wrist)"
        )
    
    # Process gun events
    # Group consecutive gun detections to avoid spam
    if gun_events:
        gun_by_person = {}
        for event in gun_events:
            person_idx = event.get('person_index', 0)
            if person_idx not in gun_by_person:
                gun_by_person[person_idx] = []
            gun_by_person[person_idx].append(event)
        
        for person_idx, events in gun_by_person.items():
            # Just report the first detection time for each person
            first_event = events[0]
            gun_time = first_event['frame'] / fps
            hand = first_event['hand']
            # Try to map person_idx to track_id (might not be perfect)
            # For now, we'll just use person_idx as a fallback
            justifications.append(
                f"Gun held in {hand} hand detected at {gun_time:.1f}s (person_index: {person_idx})"
            )
    
    if not justifications:
        return "No criminal activity detected."
    
    return " ".join(justifications)


def run_scene_labeling(input_dir="input", output_file="data/scene_labels.json"):
    """Run theft and gun detection on all videos and generate scene labels."""
    
    input_path = Path(input_dir)
    videos = sorted(input_path.glob("*.mp4"))
    
    if len(videos) != 4:
        print(f"Warning: Expected 4 videos, found {len(videos)}")
    
    print(f"\n{'='*60}")
    print(f"SCENE LABELING - Processing {len(videos)} videos")
    print(f"{'='*60}\n")
    
    # Load global identity catalog
    print("Loading global identity catalog...")
    track_to_global = load_global_identity_catalog()
    
    results = {
        "clips": [],
        "metadata": {
            "total_clips": len(videos),
            "crime_clips": 0,
            "normal_clips": 0
        }
    }
    
    for video in videos:
        clip_id = video.stem  # e.g., "1", "2", "3", "4"
        print(f"\n{'='*60}")
        print(f"Processing Clip {clip_id}: {video.name}")
        print(f"{'='*60}\n")
        
        # Get video FPS for timestamp calculations
        import cv2
        cap = cv2.VideoCapture(str(video))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # Define region for theft detection (you may need to adjust per video)
        # Using default from launch.json for video 1
        if clip_id == "1":
            region = (267, 40, 463, 421)
        else:
            # For other videos, you might need to define regions or skip theft detection
            region = (267, 40, 463, 421)  # Using same region as placeholder
        
        # Run theft detection
        print(f"\n--- Running Theft Detection on Clip {clip_id} ---")
        theft_output = f"data/output_videos/clip_{clip_id}_theft.mp4"
        theft_events = detect_hand_theft(
            str(video), 
            region, 
            output_path=theft_output,
            pose_conf=0.3
        )
        
        # Run gun detection
        print(f"\n--- Running Gun Detection on Clip {clip_id} ---")
        gun_output = f"data/output_videos/clip_{clip_id}_gun.mp4"
        gun_events = test_gun_detection(
            str(video),
            output_path=gun_output,
            conf_threshold=0.5,
            pose_conf=0.3
        )
        
        # Determine label
        is_crime = len(theft_events) > 0 or len(gun_events) > 0
        label = "crime" if is_crime else "normal"
        
        # Format justification
        justification = format_justification(
            theft_events or [], 
            gun_events or [], 
            clip_id, 
            track_to_global,
            fps
        )
        
        # Add to results
        results["clips"].append({
            "clip_id": clip_id,
            "label": label,
            "justification": justification,
            "theft_events_count": len(theft_events) if theft_events else 0,
            "gun_events_count": len(gun_events) if gun_events else 0
        })
        
        if is_crime:
            results["metadata"]["crime_clips"] += 1
        else:
            results["metadata"]["normal_clips"] += 1
        
        print(f"\n✅ Clip {clip_id}: {label.upper()}")
        print(f"   {justification}")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SCENE LABELING COMPLETE")
    print(f"{'='*60}")
    print(f"Total clips: {results['metadata']['total_clips']}")
    print(f"Crime clips: {results['metadata']['crime_clips']}")
    print(f"Normal clips: {results['metadata']['normal_clips']}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scene labeling with theft and gun detection")
    parser.add_argument("--input", default="input", help="Input directory with videos")
    parser.add_argument("--output", default="data/scene_labels.json", help="Output JSON file")
    
    args = parser.parse_args()
    run_scene_labeling(args.input, args.output)

