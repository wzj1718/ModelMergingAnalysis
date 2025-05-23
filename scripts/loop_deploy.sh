#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

MODEL=""
MAX_LORAS=20
ROOT=""
MAX_LORA_RANK=16

PORTS=(9112)
NUM_GPUS=${#PORTS[@]}

echo "Deploying model $MODEL with $MAX_LORAS LORAs on $NUM_GPUS GPUs"
echo "Starting API servers..."

mkdir -p vllm_logs/$ROOT

COMMON_ARGS="--model $MODEL \
    --trust-remote-code \
    --enable-lora \
    --seed 42 \
    --max-lora-rank $MAX_LORA_RANK \
    --gpu-memory-utilization 0.45 \
    --max-loras $MAX_LORAS \
    --max-cpu-loras $MAX_LORAS \
    --max-model-len 8192"

if [ ${#PORTS[@]} -lt $NUM_GPUS ]; then
    echo "Error: Not enough ports defined for $NUM_GPUS GPUs"
    exit 1
fi

for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.openai.api_server \
        $COMMON_ARGS \
        --port ${PORTS[$i]} > vllm_logs/$ROOT/port_$((i+1)).log 2>&1 &
    
    SLEEP_TIME=$((2 + i/2))
    sleep $SLEEP_TIME
    echo "API server $((i+1)) started on GPU $i, port ${PORTS[$i]}"
done

echo "Waiting for servers to start ..."
sleep 30
echo "All $NUM_GPUS API servers have been started"