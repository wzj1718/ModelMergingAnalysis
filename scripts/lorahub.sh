export TOKENIZERS_PARALLELISM=true

ITERS=50
ROOT_DIR="input your root dir here"
MODEL_NAME="input your model name here"

TASKS='flores101'
TEST_TASKS='flores101'

WEIGHTS='1.0'
COMBINE_METHOD=linear

SEEDS=(41 42 47 53 3407)

for SEED in "${SEEDS[@]}"; do
    echo "Running LoRAHub with seed: $SEED"
    python run_lorahub.py \
        --N 10 \
        --max_iter $ITERS \
        --tasks $TASKS \
        --test_tasks $TEST_TASKS \
        --task_weights $WEIGHTS \
        --model_path /$ROOT_DIR/models/$MODEL_NAME \
        --lora_dir /$ROOT_DIR/lora/$MODEL_NAME/lora \
        --seed $SEED \
        --early_stop_iter 5 \
        --plot_enabled \
        --combine_method $COMBINE_METHOD \
        --ports 9112 \
        --do_search

    echo "Process with seed $SEED completed, wait 10 seconds"
    sleep 10
done