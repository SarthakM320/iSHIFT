# !/bin/bash

NUM_GPUS=8
SCRIPT="inference/multi_gpu_inference.py"

TEST_NAMES=("train")

for TEST_NAME in "${TEST_NAMES[@]}"; do
    TEST_FILE="sample_dataset/${TEST_NAME}.json"
    OUTPUT_DIR="inference/results/${TEST_NAME}"

    mkdir -p "$OUTPUT_DIR"

    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        echo "Starting $TEST_NAME on GPU $GPU_ID..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT --gpu_id $GPU_ID --num_gpus $NUM_GPUS --test_file $TEST_FILE --output_dir $OUTPUT_DIR &
    done

    echo "All jobs for $TEST_NAME started. Waiting for completion..."
    wait
    echo "All jobs for $TEST_NAME completed!"

    # Merge results
    echo "Merging results for $TEST_NAME..."
    cat $OUTPUT_DIR/results_gpu*.jsonl > $OUTPUT_DIR/results_all.jsonl
    echo "Merged results saved to $OUTPUT_DIR/results_all.jsonl"
done