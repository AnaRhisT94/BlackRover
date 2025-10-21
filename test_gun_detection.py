#!/usr/bin/env python3
"""Critical gun detection - only detects guns held in hands."""

import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path


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


def check_gun_in_hand(gun_bbox, wrist_positions, hand_radius=30, iou_threshold=0.2):
    """Check if gun intersects with hand/wrist (CRITICAL threat only)."""
    if wrist_positions is None:
        return None
    
    left_wrist, right_wrist = wrist_positions
    
    for hand_name, wrist in [("left", left_wrist), ("right", right_wrist)]:
        if wrist is not None and len(wrist) == 2:
            hand_bbox = [
                wrist[0] - hand_radius, wrist[1] - hand_radius,
                wrist[0] + hand_radius, wrist[1] + hand_radius
            ]
            
            iou = compute_iou(gun_bbox, hand_bbox)
            if iou > iou_threshold:
                return hand_name
    
    return None


def test_gun_detection(video_path, output_path=None, conf_threshold=0.5, pose_conf=0.3):
    """Detect CRITICAL threats only (guns held in hands) - PERSON-CENTRIC APPROACH."""
    print("Loading models...")
    model = YOLO('best.pt')
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
        output_path = f"data/output_videos/{video_name}_critical.mp4"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"Processing {total_frames} frames...")
    print(f"Method: PERSON-CENTRIC (detecting guns within person bounding boxes only)")
    
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10
    
    frame_count = 0
    critical_threats = []
    consecutive_critical = 0
    max_consecutive = 0
    is_crime = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # First detect persons with pose
        pose_results = pose_model(frame, conf=pose_conf, verbose=False)
        
        # Extract person bounding boxes and wrist positions
        person_data = []
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            pose_result = pose_results[0]
            if pose_result.keypoints.xy is not None and pose_result.keypoints.conf is not None:
                keypoints = pose_result.keypoints.xy.cpu().numpy()
                keypoint_conf = pose_result.keypoints.conf.cpu().numpy()
                
                # Get person bounding boxes from pose detection
                person_boxes = []
                if pose_result.boxes is not None and pose_result.boxes.xyxy is not None:
                    person_boxes = pose_result.boxes.xyxy.cpu().numpy()
                
                for i in range(len(keypoints)):
                    left_wrist = None
                    right_wrist = None
                    
                    if keypoint_conf[i, LEFT_WRIST_IDX] >= 0.3:
                        left_wrist = tuple(keypoints[i, LEFT_WRIST_IDX])
                    if keypoint_conf[i, RIGHT_WRIST_IDX] >= 0.3:
                        right_wrist = tuple(keypoints[i, RIGHT_WRIST_IDX])
                    
                    # Get person bbox if available
                    person_bbox = None
                    if i < len(person_boxes):
                        person_bbox = person_boxes[i]
                    
                    person_data.append({
                        'wrists': (left_wrist, right_wrist),
                        'bbox': person_bbox,
                        'index': i
                    })
        
        # Check for CRITICAL threats - RUN GUN DETECTION ON EACH PERSON PATCH
        frame_is_critical = False
        for person in person_data:
            if person['bbox'] is None:
                continue
            
            # Extract person bounding box
            px1, py1, px2, py2 = map(int, person['bbox'])
            
            # Add margin around person
            margin = 20
            px1 = max(0, px1 - margin)
            py1 = max(0, py1 - margin)
            px2 = min(width, px2 + margin)
            py2 = min(height, py2 + margin)
            
            # Crop person region
            person_crop = frame[py1:py2, px1:px2]
            
            if person_crop.size == 0:
                continue
            
            # Run gun detection on person crop only
            person_results = model(person_crop, verbose=False)
            
            # Check if gun detected in this person's region
            for result in person_results:
                for pos, detection in enumerate(result.boxes.xyxy):
                    if result.boxes.conf[pos] >= conf_threshold:
                        class_name = result.names[int(result.boxes.cls[pos])]
                        
                        if 'gun' in class_name.lower() or 'pistol' in class_name.lower():
                            # Convert crop coordinates back to full frame coordinates
                            gun_bbox_crop = detection.cpu().numpy()
                            gun_bbox = [
                                int(gun_bbox_crop[0] + px1),
                                int(gun_bbox_crop[1] + py1),
                                int(gun_bbox_crop[2] + px1),
                                int(gun_bbox_crop[3] + py1)
                            ]
                            
                            # Check if gun is in hand
                            hand = check_gun_in_hand(gun_bbox, person['wrists'])
                            if hand:
                                # CRITICAL threat found
                                critical_threats.append({
                                    'frame': frame_count,
                                    'class': class_name,
                                    'confidence': float(result.boxes.conf[pos]),
                                    'bbox': gun_bbox,
                                    'hand': hand,
                                    'person_index': person['index']
                                })
                                
                                print(f"🚨 CRITICAL Frame {frame_count}: {class_name} in {hand} hand (conf: {result.boxes.conf[pos]:.2f}) [Person {person['index']}]")
                                
                                # Draw CRITICAL threat
                                cv2.rectangle(frame, (gun_bbox[0], gun_bbox[1]), (gun_bbox[2], gun_bbox[3]), (0, 0, 255), 3)
                                label = f"CRITICAL: {class_name} {result.boxes.conf[pos]:.2f}"
                                cv2.putText(frame, label, (gun_bbox[0], gun_bbox[1] - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                frame_is_critical = True
                                break
        
        # Track consecutive critical frames
        if frame_is_critical:
            consecutive_critical += 1
            max_consecutive = max(max_consecutive, consecutive_critical)
            if consecutive_critical >= 15 and not is_crime:
                is_crime = True
                print(f"\n🚨🚨🚨 CRIME DETECTED at frame {frame_count} 🚨🚨🚨\n")
        else:
            consecutive_critical = 0
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"\n{'='*40}")
    print(f"CRITICAL THREATS: {len(critical_threats)}")
    print(f"Max consecutive critical frames: {max_consecutive}")
    print(f"CLASSIFICATION: {'CRIME' if is_crime else 'NO CRIME'}")
    print(f"{'='*40}")
    print(f"Output: {output_path}")
    
    return critical_threats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect CRITICAL threats (guns held in hands)")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("-c", "--conf", type=float, default=0.5, help="Gun detection confidence (default: 0.5)")
    parser.add_argument("-p", "--pose-conf", type=float, default=0.3, help="Pose confidence (default: 0.3)")
    
    args = parser.parse_args()
    test_gun_detection(args.video, args.output, args.conf, args.pose_conf)

