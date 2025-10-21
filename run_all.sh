#!/bin/bash
# Master script to reproduce all outputs for surveillance video analysis
# This script runs all 4 pipeline steps in sequence

set -e  # Exit on any error

echo "=========================================="
echo "Surveillance Video Analysis Pipeline"
echo "=========================================="
echo ""

# Check if input videos exist
if [ ! -d "input" ] || [ -z "$(ls -A input/*.mp4 2>/dev/null)" ]; then
    echo "❌ Error: No video files found in input/ directory"
    echo "Please place 1.mp4, 2.mp4, 3.mp4, 4.mp4 in the input/ folder"
    exit 1
fi

echo "✅ Found input videos:"
ls -1 input/*.mp4
echo ""

# Step 1: Build Person Galleries
echo "=========================================="
echo "STEP 1/4: Building Person Galleries"
echo "=========================================="
echo "This will process all videos and extract person tracks with embeddings..."
echo ""

python main.py \
    --input input/ \
    --enable-tracking \
    --build-gallery \
    --device cuda \
    --skip-frames 1

echo ""
echo "✅ Step 1 complete: Person galleries saved to data/person_gallery/"
echo ""

# Step 2: Create Global Identity Catalog
echo "=========================================="
echo "STEP 2/4: Creating Global Identity Catalog"
echo "=========================================="
echo "Matching persons across videos using embedding similarity..."
echo ""

python create_global_identity_catalog.py

echo ""
echo "✅ Step 2 complete: Global catalog saved to data/person_gallery/global_identity_catalog.json"
echo ""

# Step 3: Render Videos with Global IDs
echo "=========================================="
echo "STEP 3/4: Rendering Videos with Global IDs"
echo "=========================================="
echo "Creating annotated videos with global person IDs..."
echo ""

python render_videos_with_global_ids.py

echo ""
echo "✅ Step 3 complete: Annotated videos saved to data/output_videos_global_ids/"
echo ""

# Step 4: Scene Labeling (Theft + Gun Detection)
echo "=========================================="
echo "STEP 4/4: Scene Labeling (Crime Detection)"
echo "=========================================="
echo "Running theft and gun detection on all videos..."
echo ""

python run_scene_labeling.py --input input/ --output data/scene_labels.json

echo ""
echo "✅ Step 4 complete: Scene labels saved to data/scene_labels.json"
echo ""

# Final Summary
echo "=========================================="
echo "🎉 PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "📦 Deliverables Generated:"
echo ""
echo "Part A: Person Identity Catalogue"
echo "  ✅ data/person_gallery/global_identity_catalog.json"
echo "  ✅ data/person_gallery/track_to_global_id_mapping.json"
echo ""
echo "Part B: Scene Labels"
echo "  ✅ data/scene_labels.json"
echo ""
echo "Visualizations:"
echo "  ✅ data/person_gallery/{1,2,3,4}/gallery_grid.jpg"
echo "  ✅ data/output_videos_global_ids/video_*_with_global_ids.mp4"
echo ""
echo "=========================================="
echo ""
echo "To view results:"
echo "  - Person catalog: cat data/person_gallery/global_identity_catalog.json | jq"
echo "  - Scene labels: cat data/scene_labels.json | jq"
echo "  - Gallery grids: open data/person_gallery/*/gallery_grid.jpg"
echo "  - Videos: open data/output_videos_global_ids/*.mp4"
echo ""

