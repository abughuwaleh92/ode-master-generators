# scripts/batch_verify.sh
#!/bin/bash
# Batch verification with retry logic

# Benefits:
# - Handles large datasets in chunks
# - Automatic retry for failed verifications
# - Progress tracking
# - Memory-efficient processing

DATASET=$1
CHUNK_SIZE=${2:-100}
MAX_RETRIES=${3:-3}

if [ -z "$DATASET" ]; then
    echo "Usage: $0 <dataset.jsonl> [chunk_size] [max_retries]"
    exit 1
fi

echo "Starting batch verification..."
echo "Dataset: $DATASET"
echo "Chunk size: $CHUNK_SIZE"

# Split dataset into chunks
split -l $CHUNK_SIZE "$DATASET" chunk_

# Process each chunk
for chunk in chunk_*; do
    echo "Processing $chunk..."
    
    retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if python verify_chunk.py "$chunk" "${chunk}_verified.jsonl"; then
            echo "✓ $chunk verified successfully"
            break
        else
            retry_count=$((retry_count + 1))
            echo "✗ $chunk failed, retry $retry_count/$MAX_RETRIES"
            sleep 5
        fi
    done
done

# Merge verified chunks
cat chunk_*_verified.jsonl > "${DATASET%.jsonl}_verified.jsonl"

# Cleanup
rm chunk_*

echo "Verification complete!"