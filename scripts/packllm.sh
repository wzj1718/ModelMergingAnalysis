export TOKENIZERS_PARALLELISM=true

ROOT_DIR=
MODEL_NAME=gemma-2-2b-it

TASKS='flores101'
TEST_TASKS='flores101'
WEIGHTS='1.0'
METHOD=simple
COMBINE_METHOD=linear
TOPK=10
TAO=0.1

SEEDS=(42)
sleep 30

for SEED in "${SEEDS[@]}"; do
    echo "Running PackLLM with seed: $SEED"
    python run_packllm.py \
        --N 10 \
        --method $METHOD \
        --tasks $TASKS \
        --test_tasks $TEST_TASKS \
        --tao $TAO \
        --topK $TOPK \
        --task_weights $WEIGHTS \
        --model_path /$ROOT_DIR/models/$MODEL_NAME \
        --lora_dir /$ROOT_DIR/lora/$MODEL_NAME/lora \
        --seed $SEED \
        --early_stop_iter 5 \
        --plot_enabled \
        --combine_method $COMBINE_METHOD \
        --ports 9112 36048 22246 13732

    echo "Process with seed $SEED completed, wait 10 seconds"
    sleep 10
done