#!/bin/bash

# Example usage of optimizer analyzer for different methods and tasks

ROOT_DIR=""
TASK='math'
METHOD=genomeplus
N=10
MODEL=gemma-2-2b-it
# MODEL=Meta-Llama-3.1-8B-Instruct
COMBINE_METHOD=linear

# Add project root to PYTHONPATH
export PYTHONPATH=$ROOT_DIR/GENOME:$PYTHONPATH

# Create base output directory
OUTPUT_DIR="analysis_results/${METHOD}_n${N}_${TASK}"

python src/analysis/run_analysis.py \
    --base_dir $ROOT_DIR/GENOME \
    --method $METHOD \
    --task_name $TASK \
    --population_size $N \
    --combine_method $COMBINE_METHOD \
    --model $MODEL \
    --output_dir $OUTPUT_DIR