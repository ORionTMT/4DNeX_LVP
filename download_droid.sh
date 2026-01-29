#!/bin/bash
set -uo pipefail

source ~/.bashrc
source /vast/projects/jgu32/lab/mutian/miniconda3/etc/profile.d/conda.sh

conda activate 4dnex


echo "=========================================="
echo "Script started at: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

datas=(
    "droid_1.0.1"
)

for data in ${datas[@]}; do
    echo ""
    echo "=========================================="
    echo "Starting download for: $data"
    echo "Time: $(date)"
    echo "=========================================="
    
    max_retries=1000
    retry_count=0
    success=false
    
    while [ $retry_count -lt $max_retries ] && [ "$success" = false ]; do
        if [ $retry_count -gt 0 ]; then
            echo ""
            echo "Retry attempt $retry_count at $(date)"
        fi
        
        echo "Starting download command at $(date)..."
        # Run command directly to see real-time output (allow failures for retry loop)
        hf download --repo-type dataset cadene/$data \
            --local-dir data/$data \
            --exclude "data/**" \
            --exclude "**/observation.images.wrist_left/**" \
            --max-workers 4
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo ""
            echo "=========================================="
            echo "Successfully downloaded $data at $(date)"
            echo "=========================================="
            success=true
        else
            retry_count=$((retry_count + 1))
            echo ""
            echo "=========================================="
            echo "Error encountered while downloading $data (exit code: $exit_code)"
            echo "Waiting 5 minutes before retry (attempt $retry_count/$max_retries)..."
            echo "Time: $(date)"
            echo "=========================================="
            sleep 30
        fi
    done
    
    if [ "$success" = false ]; then
        echo ""
        echo "=========================================="
        echo "Failed to download $data after $max_retries retries"
        echo "Time: $(date)"
        echo "=========================================="
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All downloads completed successfully!"
echo "Script finished at: $(date)"
echo "=========================================="
