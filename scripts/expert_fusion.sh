export TOKENIZERS_PARALLELISM=true

ITERS=50
ROOT_DIR="input your root dir here"
MODEL_NAME=gemma-2-2b-it
LORA_PATH="input your lora path here"

TASKS='flores101'
TEST_TASKS='flores101'
WEIGHTS='1.0'
COMBINE_METHOD=linear

SEEDS=(42)

for SEED in "${SEEDS[@]}"; do
    echo "Running Expert Fusion with seed: $SEED"
    python run_lorahub.py \
        --N 10 \
        --max_iter $ITERS \
        --tasks $TASKS \
        --test_tasks $TEST_TASKS \
        --task_weights $WEIGHTS \
        --model_path /$ROOT_DIR/models/$MODEL_NAME \
        --lora_dir $LORA_PATH \
        --seed $SEED \
        --early_stop_iter 5 \
        --plot_enabled \
        --combine_method $COMBINE_METHOD \
        --ports 9112

    echo "Process with seed $SEED completed, wait 10 seconds"
    sleep 10
done