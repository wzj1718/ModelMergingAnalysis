export TOKENIZERS_PARALLELISM=true

N=10
ITERS=10

ROOT_DIR=""
MODEL_NAME=""
LORA_PATH=""


TASKS="drop"
TEST_TASKS='drop'
WEIGHTS='1.0'
COMBINE_METHOD=dare_ties_magnitude_with_rescaler  
CROSS_METHOD=dare_ties_magnitude_with_rescaler 

SEEDS=(185 3013 45 111 63 49 89 135 42 66 596)
for SEED in "${SEEDS[@]}"; do
    echo "Running GENOME with seed: $SEED"
    python run_genome.py \
        --tasks $TASKS \
        --test_tasks $TEST_TASKS \
        --task_weights $WEIGHTS \
        --model_path $ROOT_DIR/$MODEL_NAME \
        --lora_dir $ROOT_DIR/$MODEL_NAME/lora/$MODEL_NAME \

        --population_size $N \
        --combine_method $COMBINE_METHOD \
        --cross_method $CROSS_METHOD \
        --plot_enabled \
        --iters $ITERS \
        --seed $SEED \
        --cross_rate 0.3 \
        --individual_mutation_rate 0.1 \
        --gene_mutation_rate 0.1 \
        --sigma 0.001 \
        --elite_percent 0.1 \
        --early_stop_iter 5 \
        --ports 7789
    
    echo "Process with seed $SEED completed, wait 10 seconds"
    sleep 10
done