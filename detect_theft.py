#!/usr/bin/env python3
"""
Hand/Wrist Theft Detection
Tracks hand movement: inside region -> outside -> to stomach area = theft
"""

import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
from collections import defaultdict
import time


def compute_iou(box1, box2):
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


class WristTracker:
    """Track individual wrist movement through theft region"""
    
    def __init__(self, wrist_id, person_id, region_bbox, fps):
        self.wrist_id = wrist_id
        self.person_id = person_id
        self.region_bbox = region_bbox  # (x1, y1, x2, y2)
        self.fps = fps
        self.is_inside_now = False
        self.was_inside_recently = False
        self.last_inside_frame = None
        self.last_outside_frame = None
        self.theft_detected = False
        self.theft_frame = None
        self.theft_cooldown = 0  # Frames to wait before detecting again
        self.max_time_to_stomach = int(fps * 1.0)  # Max 1.0 seconds from exit to stomach
        self.consecutive_stomach_frames = 0  # Count frames in stomach
        self.min_stomach_frames = 15  # Require 15 frames in stomach (~0.5s at 30fps)
        self.wrist_radius = 30  # For creating bounding box around wrist
        
    def is_inside_region(self, wrist_pos):
        """Check if wrist is inside the target region"""
        x, y = wrist_pos
        x1, y1, x2, y2 = self.region_bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def get_wrist_bbox(self, wrist_pos):
        """Get bounding box around wrist position"""
        x, y = wrist_pos
        r = self.wrist_radius
        return [x - r, y - r, x + r, y + r]
    
    def check_stomach_area(self, wrist_pos, person_bbox):
        """Check if wrist is in stomach area (middle-lower portion of person bbox)"""
        if not wrist_pos or person_bbox is None:
            return False
        
        x, y = wrist_pos
        px1, py1, px2, py2 = person_bbox
        
        # Calculate stomach area (middle 60% horizontally, 35-75% vertically) - bigger area
        height = py2 - py1
        width = px2 - px1
        
        stomach_x1 = px1 + width * 0.2
        stomach_x2 = px2 - width * 0.2
        stomach_y1 = py1 + height * 0.35
        stomach_y2 = py1 + height * 0.75
        
        # Check if wrist is in stomach area
        return stomach_x1 <= x <= stomach_x2 and stomach_y1 <= y <= stomach_y2
    
    def update(self, frame_num, wrist_pos, person_bbox):
        """Update tracker with new wrist position and person bbox"""
        # If wrist is lost, keep previous state (don't reset)
        if wrist_pos is None:
            return None
        
        is_in_region = self.is_inside_region(wrist_pos)
        is_in_stomach = self.check_stomach_area(wrist_pos, person_bbox) if person_bbox else False
        status_msg = None
        
        # Decrease theft cooldown
        if self.theft_cooldown > 0:
            self.theft_cooldown -= 1
        
        # INSTANT tracking: INSIDE or OUTSIDE region (no frame confirmation)
        if is_in_region:
            self.is_inside_now = True
            self.was_inside_recently = True
            self.last_inside_frame = frame_num
            # Clear previous theft detection when entering region again
            if self.theft_detected:
                self.theft_detected = False
        else:
            self.is_inside_now = False
            self.last_outside_frame = frame_num
            
            # Check if too much time has passed since exiting region
            if self.was_inside_recently and self.last_outside_frame and self.last_inside_frame:
                frames_since_inside = frame_num - self.last_inside_frame
                if frames_since_inside > self.max_time_to_stomach:
                    # Too long ago, reset
                    self.was_inside_recently = False
        
        # Count consecutive frames in stomach
        if self.was_inside_recently and not self.is_inside_now and is_in_stomach and not self.theft_detected:
            self.consecutive_stomach_frames += 1
        else:
            self.consecutive_stomach_frames = 0
        
        # Detect theft: was inside recently → now outside → in stomach for min frames (within time window)
        if (self.was_inside_recently and not self.is_inside_now 
            and self.consecutive_stomach_frames >= self.min_stomach_frames
            and not self.theft_detected and self.theft_cooldown == 0):
            # Check time constraint
            if self.last_inside_frame and self.last_outside_frame:
                frames_since_inside = frame_num - self.last_inside_frame
                if frames_since_inside <= self.max_time_to_stomach:
                    self.theft_detected = True
                    self.theft_frame = frame_num
                    self.theft_cooldown = int(self.fps * 5)  # 5 second cooldown
                    status_msg = f"[Person {self.person_id}] {self.wrist_id.upper()}: 🚨 THEFT at frame {frame_num}"
                    # Reset for next detection
                    self.was_inside_recently = False
                    self.consecutive_stomach_frames = 0
        
        return status_msg


def detect_hand_theft(video_path, region_bbox, output_path=None, pose_conf=0.3, 
                     obj_conf=0.3, bag_iou_threshold=0.1):
    """
    Detect theft by tracking hand/wrist movement through a region and back to body.
    
    Args:
        video_path: Path to video file
        region_bbox: (x1, y1, x2, y2) - the target region (shelf/drawer)
        output_path: Output video path
        pose_conf: Confidence threshold for pose detection
        obj_conf: Not used (kept for backwards compatibility)
        bag_iou_threshold: Not used (kept for backwards compatibility)
    """
    print("Loading YOLO model...")
    pose_model = YOLO('yolo11n-pose.pt')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"data/output_videos/{video_name}_theft_detection.mp4"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"\n{'='*60}")
    print(f"THEFT DETECTION - Starting Processing")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Target region: {region_bbox}")
    print(f"Detection method: Hand to body movement")
    print(f"{'='*60}\n")
    
    # YOLO pose keypoint indices
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10
    
    # Track each detected person's wrists
    person_trackers = defaultdict(lambda: {
        'left': None,
        'right': None
    })
    
    frame_count = 0
    theft_events = []
    
    x1, y1, x2, y2 = region_bbox
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw the target region
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, f"TARGET REGION [{x1},{y1}]->[{x2},{y2}]", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Detect poses WITH TRACKING (persist=True keeps IDs stable across frames)
        pose_results = pose_model.track(frame, conf=pose_conf, verbose=False, persist=True)
        
        frame_has_theft = False
        
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            pose_result = pose_results[0]
            if pose_result.keypoints.xy is not None and pose_result.keypoints.conf is not None:
                keypoints = pose_result.keypoints.xy.cpu().numpy()
                keypoint_conf = pose_result.keypoints.conf.cpu().numpy()
                
                # Get person bounding boxes
                person_bboxes = []
                if pose_result.boxes is not None and pose_result.boxes.xyxy is not None:
                    boxes = pose_result.boxes.xyxy.cpu().numpy()
                    person_bboxes = [tuple(map(int, box)) for box in boxes]
                
                # Get stable track IDs from YOLO tracking
                track_ids = []
                if pose_result.boxes is not None and pose_result.boxes.id is not None:
                    track_ids = pose_result.boxes.id.cpu().numpy().astype(int)
                else:
                    # Fallback to array indices if tracking not available
                    track_ids = list(range(len(keypoints)))
                
                for idx in range(len(keypoints)):
                    track_id = track_ids[idx]  # Use stable track ID instead of array index
                    # Get wrist positions (use idx for array access)
                    left_wrist = None
                    right_wrist = None
                    
                    if keypoint_conf[idx, LEFT_WRIST_IDX] >= pose_conf:
                        left_wrist = tuple(map(int, keypoints[idx, LEFT_WRIST_IDX]))
                    if keypoint_conf[idx, RIGHT_WRIST_IDX] >= pose_conf:
                        right_wrist = tuple(map(int, keypoints[idx, RIGHT_WRIST_IDX]))
                    
                    # Only process people with at least one detected wrist
                    if left_wrist is None and right_wrist is None:
                        continue
                    
                    # Get person bbox for this person (use idx for array access)
                    person_bbox = person_bboxes[idx] if idx < len(person_bboxes) else None
                    
                    # Draw person bbox and stomach area (only for tracked people)
                    if person_bbox:
                        px1, py1, px2, py2 = person_bbox
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (128, 255, 128), 2)
                        cv2.putText(frame, f"ID {track_id}", (px1, py1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 255, 128), 2)
                        
                        # Draw stomach area with bright highlight (bigger area)
                        height = py2 - py1
                        width = px2 - px1
                        stomach_x1 = int(px1 + width * 0.2)
                        stomach_x2 = int(px2 - width * 0.2)
                        stomach_y1 = int(py1 + height * 0.35)
                        stomach_y2 = int(py1 + height * 0.75)
                        # Bright cyan rectangle with thick border
                        cv2.rectangle(frame, (stomach_x1, stomach_y1), (stomach_x2, stomach_y2), 
                                    (255, 255, 0), 3)  # Bright cyan, thick border
                        cv2.putText(frame, "STOMACH AREA", (stomach_x1 + 5, stomach_y1 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    # Initialize trackers if needed (use track_id for dictionary key)
                    if person_trackers[track_id]['left'] is None:
                        person_trackers[track_id]['left'] = WristTracker('left_wrist', track_id, region_bbox, fps)
                    if person_trackers[track_id]['right'] is None:
                        person_trackers[track_id]['right'] = WristTracker('right_wrist', track_id, region_bbox, fps)
                    
                    # Update trackers and get status messages
                    trackers = person_trackers[track_id]
                    
                    for wrist_name, wrist_pos in [('left', left_wrist), ('right', right_wrist)]:
                        if wrist_pos:
                            tracker = trackers[wrist_name]
                            status_msg = tracker.update(frame_count, wrist_pos, person_bbox)
                            
                            if status_msg:
                                print(status_msg)
                            
                            # Visualize wrists - simple circles only
                            if tracker.is_inside_now:
                                color = (0, 165, 255)  # Orange - inside
                                cv2.circle(frame, wrist_pos, 8, color, -1)
                            else:
                                color = (0, 255, 0)  # Green - outside
                                cv2.circle(frame, wrist_pos, 6, color, -1)
                            
                            # Show THEFT detection near person bbox (for 2 seconds only)
                            if tracker.theft_detected and tracker.theft_frame and person_bbox:
                                frames_since_theft = frame_count - tracker.theft_frame
                                display_duration = int(fps * 2)  # 2 seconds
                                
                                if frames_since_theft <= display_duration:
                                    px1, py1, px2, py2 = person_bbox
                                    # Display THEFT in red near person's body
                                    cv2.putText(frame, "THEFT", 
                                               (px1 + 10, py1 + 40),
                                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                    frame_has_theft = True
                            
                            # Record theft event (only once per detection)
                            if tracker.theft_detected and tracker.theft_frame:
                                # Only record once for this specific theft event
                                if not any(e['track_id'] == track_id and e['wrist'] == wrist_name 
                                         and e['theft_frame'] == tracker.theft_frame 
                                         for e in theft_events):
                                    theft_events.append({
                                        'track_id': track_id,
                                        'wrist': wrist_name,
                                        'last_inside_frame': tracker.last_inside_frame,
                                        'theft_frame': tracker.theft_frame
                                    })
        
        # Top-left tracker removed - only showing THEFT near person
        
        # Display theft alert banner if active
        if frame_has_theft:
            cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 255), -1)
            cv2.putText(frame, "🚨 THEFT DETECTED 🚨", (width//2 - 180, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Display frame info
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Confirmed Thefts: {len(theft_events)}", (10, height - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"[Progress] {frame_count}/{total_frames} frames ({len(theft_events)} confirmed thefts)")
    
    cap.release()
    out.release()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"THEFT DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total CONFIRMED theft events: {len(theft_events)}")
    
    if theft_events:
        print(f"\nDetailed Theft Events:")
        for i, event in enumerate(theft_events, 1):
            print(f"\n  🚨 Theft Event {i}:")
            print(f"     Track ID: {event['track_id']}")
            print(f"     Wrist: {event['wrist']}")
            print(f"     Last inside frame: {event['last_inside_frame']} ({event['last_inside_frame']/fps:.2f}s)")
            print(f"     Theft detected at: frame {event['theft_frame']} ({event['theft_frame']/fps:.2f}s)")
    else:
        print("\n  ✅ No thefts detected")
    
    print(f"\n{'='*60}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return theft_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect theft by tracking hand movement to bags")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--x1", type=int, default=267, help="Region x1 (default: 267)")
    parser.add_argument("--y1", type=int, default=40, help="Region y1 (default: 40)")
    parser.add_argument("--x2", type=int, default=463, help="Region x2 (default: 463)")
    parser.add_argument("--y2", type=int, default=421, help="Region y2 (default: 421)")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("-p", "--pose-conf", type=float, default=0.3, help="Pose confidence (default: 0.3)")
    parser.add_argument("--obj-conf", type=float, default=0.3, help="Object detection confidence (default: 0.3)")
    parser.add_argument("--bag-iou", type=float, default=0.1, help="Wrist-bag IoU threshold (default: 0.1)")
    
    args = parser.parse_args()
    
    region = (args.x1, args.y1, args.x2, args.y2)
    detect_hand_theft(args.video, region, args.output, args.pose_conf, args.obj_conf, args.bag_iou)