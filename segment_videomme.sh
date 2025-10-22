#!/bin/bash

video_dir='/home/hk-project-p0022573/lmu_xjh4853/workspace_ysk_wht/hkfswork/lmu_xjh4853-m3-agent/m3-agent-videomme/data/videos'
output_dir='/home/hk-project-p0022573/lmu_xjh4853/workspace_ysk_wht/hkfswork/lmu_xjh4853-m3-agent/m3-agent-videomme/data/clips'
mkdir -p "$output_dir"
for video in "$video_dir"/*.mp4; do
    video_name=$(basename "$video" .mp4)
    echo "Processing video: $video_name"

    mkdir -p "$output_dir/$video_name"
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video")
    duration_seconds=$(echo "$duration" | awk '{print int($1)}')
    
    segments=$((duration_seconds / 30 + 1))
    echo "  Duration: ${duration_seconds}s, Segments: $segments"
    echo "  Processing segments..."

    for ((i=0; i<segments; i++)); do
        start=$((i * 30))
        end=$(((i + 1) * 30))
        output="$output_dir/$video_name/$i.mp4"
        ffmpeg -ss $start -i "$video" -t 30 -c copy "${output}" -hide_banner -loglevel error
    done

    echo "  Completed: $video_name"
    echo ""
done
echo "All videos processed successfully!"