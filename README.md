# Surveillance Video Analysis System

A computer vision pipeline for cross-video person re-identification and crime detection (theft and weapons) in retail surveillance footage.

## 🎯 Overview

This system processes multiple surveillance video clips to:
- **Part A**: Build a global person identity catalog across all videos
- **Part B**: Label each clip as "normal" or "crime" based on detected theft or weapon activity

## 📦 Deliverables

### Part A: Person Identity Catalogue
- **File**: `data/person_gallery/global_identity_catalog.json`
- **Format**: Machine-readable JSON containing all global person IDs with their appearances across clips
- **Contents**: For each person - clip_id, frame ranges, track IDs, quality scores

### Part B: Scene Labels
- **File**: `data/scene_labels.json`
- **Format**: Machine-readable JSON with one record per clip
- **Contents**: clip_id, label (crime/normal), justification with timestamps and person IDs

### Visualizations
- **Gallery grids**: `data/person_gallery/{1,2,3,4}/gallery_grid.jpg` - Best frame per tracked person
- **Annotated videos**: `data/output_videos_global_ids/video_*_with_global_ids.mp4` - Videos with global IDs overlaid

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ VRAM for YOLO models

### Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download required models**:

Models are downloaded automatically by ultralytics and torchreid on first run, or download manually:

```bash
# YOLO models (place in project root)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n-pose.pt

# Gun detection model (trained custom model)
# Place best.pt in project root (provided with submission)

# ReID models (auto-downloaded by torchreid to ~/.cache/torch)
# - osnet_x1_0
# - osnet_ain_x1_0
# These download automatically on first run
```

3. **Prepare input videos**:
Place your 4 video clips in the `input/` directory as:
- `input/1.mp4`
- `input/2.mp4`
- `input/3.mp4`
- `input/4.mp4`

### Run Complete Pipeline

**Single command to reproduce all outputs**:
```bash
bash run_all.sh
```

This executes all 4 steps in sequence (takes ~15-30 minutes depending on GPU):
1. Build person galleries from videos
2. Create global identity catalog
3. Render videos with global IDs
4. Run scene labeling (theft + gun detection)

---

## 📋 Detailed Pipeline Steps

### Step 1: Build Person Galleries
Extract high-quality frames and embeddings for each tracked person:

```bash
python main.py \
    --input input/ \
    --enable-tracking \
    --build-gallery \
    --device cuda \
    --skip-frames 1
```

**Output**: `data/person_gallery/{1,2,3,4}/`
- Individual track directories with best frame crops
- `gallery_grid.jpg` - visualization of all tracks
- `embedding.npy` - OSNet ReID embedding per track
- `metadata.json` - track statistics

### Step 2: Create Global Identity Catalog
Match persons across videos using embedding similarity:

```bash
python create_global_identity_catalog.py
```

**Output**: 
- `data/person_gallery/global_identity_catalog.json` ✅ **Part A deliverable**
- `data/person_gallery/track_to_global_id_mapping.json` (helper)

**Method**: Hierarchical clustering with threshold=0.7 on cosine similarity of embeddings

### Step 3: Render Videos with Global IDs
Generate annotated videos showing global person IDs:

```bash
python render_videos_with_global_ids.py
```

**Output**: `data/output_videos_global_ids/video_{1-4}_with_global_ids.mp4`

### Step 4: Scene Labeling
Detect theft and gun crimes, generate labels:

```bash
python run_scene_labeling.py --input input/ --output data/scene_labels.json
```

**Output**: `data/scene_labels.json` ✅ **Part B deliverable**

---

## 🏗️ System Architecture

### Part A: Person Re-Identification

**Within-Video Tracking (DeepSORT)**:
- YOLOv11n person detection (conf ≥ 0.4)
- DeepSORT tracker with OSNet embeddings
- Track ID persistence across frames
- Quality scoring: confidence × bbox_size × isolation_score

**Cross-Video Matching**:
1. Extract best frame per track (highest quality score)
2. Compute OSNet embeddings for each person crop
3. Build distance matrix (cosine distance between all pairs)
4. Hierarchical clustering with threshold=0.7
5. Assign global IDs to clusters

**Output Format** (`global_identity_catalog.json`):
```json
{
  "global_person_001": {
    "global_id": 1,
    "appearances": [
      {
        "clip_id": "1",
        "track_id": 1,
        "frame_range": [0, 565],
        "total_frames": 561,
        "avg_confidence": 0.90,
        "quality_score": 0.72,
        "selected_frame_idx": 565
      }
    ]
  }
}
```

### Part B: Crime Detection

**Theft Detection** (`detect_theft.py`):
- Tracks hand/wrist movement through defined region
- Pattern: inside region → exit → move to stomach area
- Requires 15 consecutive frames (~0.5s) in stomach area within 1.0s of exit
- Uses YOLOv11n-pose for keypoint detection

**Gun Detection** (`test_gun_detection.py`):
- Person-centric approach: detect guns only within person bboxes
- YOLOv8 custom-trained gun detector
- Verifies gun is in hand using IoU between gun bbox and wrist position
- Classification: "crime" if ≥15 consecutive detections

**Output Format** (`scene_labels.json`):
```json
{
  "clips": [
    {
      "clip_id": "1",
      "label": "crime",
      "justification": "Theft detected at 14.4s by global_person_001 (left wrist). Gun held in right hand detected at 7.7s",
      "theft_events_count": 1,
      "gun_events_count": 48
    }
  ],
  "metadata": {
    "total_clips": 4,
    "crime_clips": 4,
    "normal_clips": 0
  }
}
```

---

## 🔬 Technical Approach

### Assumptions

**Person ReID**:
- Same person wears same clothing across all clips
- Sufficient frontal/side views available for embedding extraction
- Lighting conditions relatively consistent across videos
- People don't have identical appearances

**Crime Detection**:
- Theft involves hand entering a defined region (shelf) then moving to body
- Only guns held in hands constitute immediate threats
- Region of interest is pre-defined (can be adjusted via CLI args)
- 29-30 FPS video for timing calculations

### Limitations

**Person ReID**:
- May merge similar-looking people with similar clothing
- Sensitive to extreme pose/angle changes
- Track fragmentation during prolonged occlusions
- False negatives if person changes clothing between clips
- Clustering threshold (0.7) is fixed, not adaptive

**Crime Detection**:
- **Theft**: Fixed region coordinates (not adaptive to scene geometry)
- **Theft**: Cannot detect pocket-picking or concealment methods without body movement
- **Gun**: False positives on gun-like objects (phones, tools held similarly)
- **Gun**: Misses holstered or concealed weapons
- **Both**: No temporal context (e.g., suspicious loitering before crime)
- **Both**: Person-index in justification may not always map to global IDs correctly

### Model Details

| Component | Model | Purpose |
|-----------|-------|---------|
| Person Detection | YOLOv11n | Detect person bounding boxes |
| Pose Estimation | YOLOv11n-pose | Detect wrist keypoints for theft |
| Gun Detection | YOLOv8 (custom) | Detect firearms |
| Person ReID | OSNet-x1.0 | Generate embeddings for matching |
| Tracking | DeepSORT | Maintain consistent track IDs |

---

## 🔮 Future Improvements

**With More Time**:

1. **Adaptive Regions**: Use object detection to identify shelves/counters automatically
2. **Action Recognition**: Classify behaviors (loitering, reaching, examining items)
3. **Temporal Modeling**: LSTM/Transformer for sequence-level understanding
4. **Better Gun Detection**: Fine-tune on retail scenarios, reduce false positives
5. **Another Better Gun Detection**: I should have tried SAM as well!
6. **Multi-camera Calibration**: Handle overlapping views from multiple angles
7. **Track Re-association**: Recover identity after long occlusions
8. **Appearance Changes**: Handle clothing changes via face recognition fallback
9. **Explainable AI**: Generate detailed event timelines with confidence scores
10. **Real-time Processing**: Optimize for streaming video analysis
11. **Anomaly Detection**: Detect unusual patterns without specific crime models

**Production Considerations**:
- Model quantization for faster inference
- Cloud deployment with load balancing
- Privacy-preserving features (face blurring)
- Alert system integration
- Database backend for large-scale deployments

---

## 📁 Project Structure

```
blackrover/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_all.sh                         # Master script to reproduce outputs
│
├── main.py                            # Step 1: Build galleries
├── create_global_identity_catalog.py  # Step 2: Global catalog
├── render_videos_with_global_ids.py   # Step 3: Render videos
├── run_scene_labeling.py              # Step 4: Scene labeling
├── detect_theft.py                    # Theft detection module
├── test_gun_detection.py              # Gun detection module
│
├── config/
│   ├── detection_config.yaml          # Detection parameters
│   └── tracking_config.yaml           # Tracking parameters
│
├── src/                               # Core modules
│   ├── detection/                     # YOLO detectors
│   ├── tracking/                      # DeepSORT, ReID
│   ├── output/                        # Visualization, I/O
│   ├── ingestion/                     # Video reading
│   └── utils/                         # Helpers
│
├── models/                            # ReID model cache
├── best.pt                            # Gun detection model
├── yolo11n.pt                         # Person detection
├── yolo11n-pose.pt                    # Pose estimation
│
├── input/                             # Input videos
│   ├── 1.mp4, 2.mp4, 3.mp4, 4.mp4
│
└── data/                              # 📦 OUTPUTS
    ├── scene_labels.json              # ✅ Part B
    ├── person_gallery/
    │   ├── global_identity_catalog.json    # ✅ Part A
    │   ├── track_to_global_id_mapping.json
    │   └── {1,2,3,4}/
    │       ├── gallery_grid.jpg
    │       └── track_XXX/
    │           ├── frame_*.jpg
    │           ├── embedding.npy
    │           └── metadata.json
    └── output_videos_global_ids/
        ├── video_1_with_global_ids.mp4
        ├── video_2_with_global_ids.mp4
        ├── video_3_with_global_ids.mp4
        └── video_4_with_global_ids.mp4
```

---

## 🐛 Troubleshooting

**CUDA Out of Memory**:
- Reduce batch size in detection configs
- Use `--device cpu` (slower but works)
- Process videos one at a time

**Model Download Fails**:
- Check internet connection
- Manually download models from links above
- Ensure ~/.cache/torch has write permissions

**Poor ReID Matching**:
- Adjust clustering threshold in `create_global_identity_catalog.py`
- Check `gallery_grid.jpg` visualizations for quality
- Ensure good lighting and frontal views in videos

**False Theft Detections**:
- Adjust region coordinates with `--x1 --y1 --x2 --y2` flags
- Tune `min_stomach_frames` in `detect_theft.py`
- Check pose detection confidence threshold

---

