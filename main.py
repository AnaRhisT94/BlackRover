"""Main processing pipeline for surveillance video analysis."""
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.ingestion.video_reader import VideoReader
from src.ingestion.preprocessor import FramePreprocessor
from src.detection.yolo_detector import YOLODetector
from src.detection.pose_detector import PoseDetector
from src.tracking.deepsort_tracker import PersonTracker
from src.tracking.trajectory_manager import TrajectoryManager
from src.tracking.reid_gallery import CrossVideoReIDGallery
from src.output.video_writer import AnnotatedVideoWriter
from src.output.visualizer import Visualizer
from src.output.frame_saver import FrameSaver
from src.output.person_gallery_builder import PersonGalleryBuilder
from src.output.cross_video_gallery import CrossVideoGalleryVisualizer
from src.utils.logger import setup_logger


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def main(args, reid_gallery=None, cross_video_visualizer=None):
    """Main processing pipeline."""
    # Setup logger
    logger = setup_logger('main', log_file='data/logs/processing.log')
    logger.info(f"Starting video processing: {args.input}")
    
    # Load configurations
    detection_config = load_config('config/detection_config.yaml')
    tracking_config = load_config('config/tracking_config.yaml')
    
    # Get video name for organizing output
    video_name = Path(args.input).stem
    
    # Set current video in reid gallery if enabled
    if reid_gallery is not None:
        reid_gallery.set_current_video(video_name)
        logger.info(f"ReID gallery set for video: {video_name}")
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Video reader
    video_reader = VideoReader(
        video_path=args.input,
        skip_frames=args.skip_frames
    )
    logger.info(f"Video info: {video_reader}")
    
    # Preprocessor
    preprocessor = FramePreprocessor(
        target_size=None,  # Keep original size
        normalize=False
    )
    
    # Detectors
    person_detector = YOLODetector(
        model_path=detection_config['yolo']['model_path'],
        conf_threshold=detection_config['yolo']['conf_threshold'],
        device=args.device
    )
    logger.info(f"Loaded person detector: {detection_config['yolo']['model_path']}")
    
    pose_detector = PoseDetector(
        model_path=detection_config['pose']['model_path'],
        conf_threshold=detection_config['pose']['conf_threshold'],
        device=args.device
    )
    logger.info(f"Loaded pose detector: {detection_config['pose']['model_path']}")
    
    # Tracker (if enabled)
    if args.enable_tracking:
        logger.info("Initializing DeepSORT tracker...")
        tracker = PersonTracker(
            max_age=tracking_config['deepsort']['max_age'],
            n_init=tracking_config['deepsort']['n_init'],
            nms_max_overlap=tracking_config['deepsort']['nms_max_overlap'],
            max_cosine_distance=tracking_config['deepsort']['max_cosine_distance'],
            nn_budget=tracking_config['deepsort']['nn_budget'],
            embedder=tracking_config['deepsort']['embedder'],
            embedder_model_name=tracking_config['deepsort']['embedder_model_name'],
            embedder_gpu=(args.device == 'cuda')
        )
        trajectory_manager = TrajectoryManager(max_history=300)
        logger.info("Tracker initialized successfully")
    else:
        tracker = None
        trajectory_manager = None
    
    # Person gallery builder (if enabled)
    if args.build_gallery:
        logger.info("Initializing person gallery builder...")
        gallery_builder = PersonGalleryBuilder(
            output_dir=args.gallery_output,
            video_name=video_name,
            min_confidence=args.gallery_min_conf,
            max_iou_threshold=0.15,  # Max overlap with other people
            min_isolation_distance=50.0  # Minimum distance for best isolation score
        )
        logger.info(f"Gallery will be saved to: {args.gallery_output}/{video_name}")
    else:
        gallery_builder = None
    
    # Visualizer
    visualizer = Visualizer()
    
    # Video writer
    output_path = args.output or f"data/output_videos/{video_name}_output.mp4"
    video_writer = AnnotatedVideoWriter(
        output_path=output_path,
        fps=video_reader.fps
    )
    logger.info(f"Output will be saved to: {output_path}")
    
    # Frame savers for individual frames
    if args.save_frames:
        logger.info("Initializing frame savers...")
        unannotated_saver = FrameSaver(
            output_dir=f"data/frames/unannotated/{video_name}",
            prefix="original",
            image_format="jpg"
        )
        annotated_saver = FrameSaver(
            output_dir=f"data/frames/annotated/{video_name}",
            prefix="annotated",
            image_format="jpg"
        )
        logger.info(f"Frames will be saved to: data/frames/{{unannotated,annotated}}/{video_name}/")
    else:
        unannotated_saver = None
        annotated_saver = None
    
    # Process video
    logger.info("Processing frames...")
    frame_count = 0
    
    try:
        with tqdm(total=video_reader.total_frames, desc="Processing") as pbar:
            for frame_idx, frame in video_reader.read_frames():
                # Preprocess
                processed_frame = preprocessor.process(frame)
                
                # Detect persons
                person_detections = person_detector.detect_persons(processed_frame)
                
                # Detect poses
                keypoints, keypoint_conf, pose_boxes = pose_detector.detect_poses(processed_frame)
                
                # Update tracker if enabled
                if args.enable_tracking and len(person_detections) > 0:
                    track_ids, track_boxes, track_confs = tracker.update(person_detections, processed_frame)
                    
                    # Match against ReID gallery if enabled (forward-only ID reassignment)
                    if reid_gallery is not None and len(track_ids) > 0:
                        # Get embeddings for current tracks
                        embeddings, valid_mask = tracker.get_embeddings(track_ids)
                        
                        if len(embeddings) > 0:
                            # Get valid track IDs and confidences
                            valid_track_ids = track_ids[valid_mask]
                            valid_confidences = track_confs[valid_mask]
                            valid_boxes = track_boxes[valid_mask]
                            
                            # Match against gallery and get global IDs
                            global_ids, is_new = reid_gallery.match_batch(
                                embeddings, valid_track_ids, valid_confidences, video_name
                            )
                            
                            # Create mapping from local track ID to global ID
                            local_to_global = {}
                            for local_id, global_id in zip(valid_track_ids, global_ids):
                                local_to_global[local_id] = global_id
                            
                            # Add to cross-video visualizer if enabled
                            if cross_video_visualizer is not None:
                                for local_id, global_id, bbox, conf in zip(valid_track_ids, global_ids, valid_boxes, valid_confidences):
                                    cross_video_visualizer.add_track(
                                        track_id=local_id,
                                        global_id=global_id,
                                        video_name=video_name,
                                        frame_idx=frame_idx,
                                        bbox=bbox,
                                        frame=frame,
                                        confidence=conf
                                    )
                            
                            # Replace track IDs with global IDs where available
                            original_track_ids = track_ids.copy()
                            for i, local_id in enumerate(track_ids):
                                if local_id in local_to_global:
                                    track_ids[i] = local_to_global[local_id]
                    
                    # Match tracks with pose detections FIRST (using IoU matching)
                    track_to_pose = {}
                    if len(track_boxes) > 0 and len(pose_boxes) > 0:
                        from scipy.optimize import linear_sum_assignment
                        
                        # Calculate IoU matrix
                        iou_matrix = np.zeros((len(track_boxes), len(pose_boxes)))
                        for i, track_box in enumerate(track_boxes):
                            for j, pose_box in enumerate(pose_boxes):
                                iou_matrix[i, j] = compute_iou(track_box, pose_box)
                        
                        # Hungarian matching
                        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                        for i, j in zip(row_ind, col_ind):
                            if iou_matrix[i, j] > 0.3:  # IoU threshold
                                track_to_pose[i] = j
                    
                    # Add frames to gallery builder if enabled
                    if args.build_gallery:
                        # Convert all track boxes to list for overlap detection
                        all_boxes = [bbox for bbox in track_boxes]
                        
                        for i, (track_id, bbox, conf) in enumerate(zip(track_ids, track_boxes, track_confs)):
                            # Get keypoints for this track if matched
                            track_keypoints = None
                            track_keypoint_conf = None
                            
                            if i in track_to_pose:
                                pose_idx = track_to_pose[i]
                                track_keypoints = keypoints[pose_idx]
                                track_keypoint_conf = keypoint_conf[pose_idx]
                            
                            gallery_builder.add_frame(
                                frame_idx=frame_idx,
                                track_id=track_id,
                                bbox=bbox,
                                confidence=conf,
                                frame=frame,
                                all_boxes=all_boxes,
                                keypoints=track_keypoints,
                                keypoint_conf=track_keypoint_conf,
                                keypoint_threshold=detection_config['pose']['keypoint_threshold']
                            )
                else:
                    track_ids = np.arange(len(person_detections))
                    track_boxes = person_detections.xyxy if len(person_detections) > 0 else np.array([])
                    track_confs = person_detections.confidence if len(person_detections) > 0 else np.array([])
                    track_to_pose = {}  # Initialize empty for the else branch
                
                # Update trajectory manager with matched data
                if args.enable_tracking and len(track_ids) > 0:
                    wrist_positions = []
                    track_keypoints = []
                    track_keypoint_confs = []
                    
                    for i in range(len(track_ids)):
                        if i in track_to_pose:
                            pose_idx = track_to_pose[i]
                            track_keypoints.append(keypoints[pose_idx])
                            track_keypoint_confs.append(keypoint_conf[pose_idx])
                            
                            # Get wrist positions
                            left_wrist_idx = pose_detector.KEYPOINT_INDICES["left_wrist"]
                            right_wrist_idx = pose_detector.KEYPOINT_INDICES["right_wrist"]
                            
                            left_wrist = None
                            right_wrist = None
                            if keypoint_conf[pose_idx, left_wrist_idx] >= detection_config['pose']['keypoint_threshold']:
                                left_wrist = tuple(keypoints[pose_idx, left_wrist_idx])
                            if keypoint_conf[pose_idx, right_wrist_idx] >= detection_config['pose']['keypoint_threshold']:
                                right_wrist = tuple(keypoints[pose_idx, right_wrist_idx])
                            
                            wrist_positions.append((left_wrist, right_wrist))
                        else:
                            track_keypoints.append(None)
                            track_keypoint_confs.append(None)
                            wrist_positions.append((None, None))
                    
                    trajectory_manager.update(
                        frame_idx=frame_idx,
                        track_ids=track_ids,
                        track_boxes=track_boxes,
                        keypoints=np.array([kp for kp in track_keypoints if kp is not None]) if any(kp is not None for kp in track_keypoints) else None,
                        keypoint_confs=np.array([kc for kc in track_keypoint_confs if kc is not None]) if any(kc is not None for kc in track_keypoint_confs) else None,
                        wrist_positions=wrist_positions
                    )
                
                # Visualize
                annotated = frame.copy()
                
                # Draw person bounding boxes with track IDs
                if len(track_boxes) > 0:
                    if args.enable_tracking:
                        labels = [f"ID: {track_id}" for track_id in track_ids]
                    else:
                        labels = [f"Person {i}" for i in range(len(track_boxes))]
                    
                    # Create detections object for visualization
                    from supervision import Detections
                    viz_detections = Detections(
                        xyxy=track_boxes,
                        confidence=track_confs if len(track_confs) > 0 else None,
                        tracker_id=track_ids  # Always pass tracker_id since we always have track_ids
                    )
                    annotated = visualizer.draw_detections(annotated, viz_detections, labels)
                
                # Draw pose keypoints
                if len(keypoints) > 0:
                    annotated = visualizer.draw_keypoints(
                        annotated,
                        keypoints,
                        keypoint_conf,
                        threshold=detection_config['pose']['keypoint_threshold']
                    )
                    
                    # Draw wrists for each person
                    left_wrists, right_wrists = pose_detector.get_wrist_positions(
                        keypoints, keypoint_conf
                    )
                    
                    for left_wrist, right_wrist in zip(left_wrists, right_wrists):
                        annotated = visualizer.draw_wrists(annotated, left_wrist, right_wrist)
                
                # Add frame info
                if args.enable_tracking:
                    active_tracks = len(track_ids)
                    total_tracks = len(trajectory_manager.tracks) if trajectory_manager else 0
                    info_text = f"Frame: {frame_idx} | Active: {active_tracks} | Total Tracks: {total_tracks}"
                else:
                    info_text = f"Frame: {frame_idx} | Persons: {len(person_detections)}"
                cv2.putText(
                    annotated, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                # Save frames if requested
                if args.save_frames:
                    # Save original unannotated frame
                    unannotated_saver.save_frame(frame, frame_idx)
                    # Save annotated frame
                    annotated_saver.save_frame(annotated, frame_idx)
                
                # Write frame to video
                video_writer.write(annotated)
                frame_count += 1
                pbar.update(args.skip_frames + 1)
        
        logger.info(f"Processing complete. Processed {frame_count} frames.")
        
        # Log tracking statistics if enabled
        if args.enable_tracking and trajectory_manager:
            stats = trajectory_manager.get_statistics()
            logger.info(f"Tracking stats: Total tracks: {stats['total_tracks']}, "
                       f"Avg duration: {stats['avg_duration']:.1f} frames, "
                       f"Max duration: {stats['max_duration']} frames")
        
        # Log ReID gallery statistics if enabled
        if reid_gallery is not None:
            reid_stats = reid_gallery.get_statistics()
            logger.info(f"ReID Gallery stats: {reid_stats['total_persons']} unique persons, "
                       f"{reid_stats['total_matches']} matches, "
                       f"{reid_stats['total_new_entries']} new entries")
        
        # Save person gallery if enabled
        if args.build_gallery:
            logger.info("Saving person gallery...")
            gallery_stats = gallery_builder.get_statistics()
            logger.info(f"Gallery stats: {gallery_stats['total_tracks']} total tracks, "
                       f"{gallery_stats['tracks_with_best_frame']} with best frames, "
                       f"{gallery_stats['tracks_without_overlap']} without overlap "
                       f"({gallery_stats['overlap_percentage']:.1f}% have overlap)")
            
            # Get embedder from tracker
            embedder = tracker.tracker.embedder if tracker else None
            
            # Save gallery (filter_overlap=True means only save non-overlapping crops)
            save_summary = gallery_builder.save_gallery(embedder=embedder, filter_overlap=True)
            logger.info(f"Gallery saved: {save_summary['saved_tracks']} tracks saved, "
                       f"{save_summary['skipped_overlap']} tracks skipped (overlap)")
            logger.info(f"Gallery location: {args.gallery_output}/{video_name}")
            logger.info(f"Grid image: {args.gallery_output}/{video_name}/gallery_grid.jpg")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        video_reader.release()
        video_writer.release()
        
        # Report frame saving
        if args.save_frames:
            logger.info(f"Saved {unannotated_saver.get_saved_count()} unannotated frames")
            logger.info(f"Saved {annotated_saver.get_saved_count()} annotated frames")
        
        logger.info("Resources released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surveillance video processing pipeline")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input video path or directory containing videos")
    parser.add_argument("--output", "-o", 
                       help="Output directory for processed videos (optional)")
    parser.add_argument("--skip-frames", type=int, default=0, 
                       help="Number of frames to skip (0 = process all)")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save individual frames (both original and annotated)")
    parser.add_argument("--enable-tracking", action="store_true",
                       help="Enable DeepSORT tracking for persistent person IDs")
    parser.add_argument("--build-gallery", action="store_true",
                       help="Build person gallery for cross-video re-identification (requires --enable-tracking)")
    parser.add_argument("--gallery-output", default="data/person_gallery",
                       help="Output directory for person gallery")
    parser.add_argument("--gallery-min-conf", type=float, default=0.7,
                       help="Minimum confidence for gallery frames")
    parser.add_argument("--enable-reid", action="store_true",
                       help="Enable cross-video re-identification (requires --enable-tracking)")
    parser.add_argument("--reid-threshold", type=float, default=0.7,
                       help="Similarity threshold for ReID matching")
    parser.add_argument("--reid-output", default="data/reid_gallery",
                       help="Output directory for ReID gallery")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.build_gallery and not args.enable_tracking:
        parser.error("--build-gallery requires --enable-tracking to be enabled")
    
    if args.enable_reid and not args.enable_tracking:
        parser.error("--enable-reid requires --enable-tracking to be enabled")
    
    # Initialize ReID gallery if enabled
    reid_gallery = None
    cross_video_visualizer = None
    if args.enable_reid:
        reid_gallery = CrossVideoReIDGallery(
            similarity_threshold=args.reid_threshold,
            output_dir=args.reid_output
        )
        cross_video_visualizer = CrossVideoGalleryVisualizer(
            output_dir=args.reid_output
        )
        print(f"ReID gallery initialized with threshold {args.reid_threshold}")
    
    # Check if input is a directory or a single file
    input_path = Path(args.input)
    if input_path.is_dir():
        # Process all videos in directory
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = [f for f in input_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in video_extensions]
        
        if not video_files:
            print(f"No video files found in directory: {args.input}")
            exit(1)
            
        print(f"Found {len(video_files)} video(s) to process")
        for video_file in video_files:
            print(f"\nProcessing: {video_file.name}")
            args.input = str(video_file)
            main(args, reid_gallery, cross_video_visualizer)
        
        # Save ReID gallery and visualizations after processing all videos
        if reid_gallery is not None:
            gallery_path = reid_gallery.save_gallery()
            print(f"\nReID gallery saved to: {gallery_path}")
            print(f"Final ReID statistics: {reid_gallery.get_statistics()}")
        
        if cross_video_visualizer is not None:
            print(f"\nGenerating cross-video visualization...")
            grid_path = cross_video_visualizer.generate_grid()
            metadata_path = cross_video_visualizer.save_metadata()
            print(f"Cross-video statistics: {cross_video_visualizer.get_statistics()}")
            print(f"Grid saved to: {grid_path}")
            print(f"Metadata saved to: {metadata_path}")
    else:
        # Process single video
        main(args, reid_gallery, cross_video_visualizer)
        
        # Save ReID gallery and visualizations after processing
        if reid_gallery is not None:
            gallery_path = reid_gallery.save_gallery()
            print(f"\nReID gallery saved to: {gallery_path}")
            print(f"Final ReID statistics: {reid_gallery.get_statistics()}")
        
        if cross_video_visualizer is not None:
            print(f"\nGenerating cross-video visualization...")
            grid_path = cross_video_visualizer.generate_grid()
            metadata_path = cross_video_visualizer.save_metadata()
            print(f"Cross-video statistics: {cross_video_visualizer.get_statistics()}")
            print(f"Grid saved to: {grid_path}")
            print(f"Metadata saved to: {metadata_path}")

