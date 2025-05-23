export TOKENIZERS_PARALLELISM=true


ROOT_DIR=""
MODEL_NAME="gemma-2-2b-it"
LORA_PATH=""


TASKS="math"
TEST_TASKS="math"
WEIGHTS="1.0"

N_INIT=10
N_MAX=10

ITERS=10

COMBINE_METHOD=linear
CROSS_METHOD=linear
METHOD=roulette
SELECTION_METHOD=roulette
SEEDS=(41 42 47 53 3407)

for SEED in "${SEEDS[@]}"; do
    echo "Running GenomePlus with seed: $SEED"
    python run_genomeplus.py \
        --model_path $MODEL_PATH \
        --lora_dir $LORA_PATH \
        --tasks $TASKS \
        --test_tasks $TEST_TASKS \
        --task_weights $WEIGHTS \
        --init_population_size $N_INIT \
        --max_population_size $N_MAX \
        --max_iter $ITERS \
        --lambda_step 0.5 \
        --phi_lambda 0.95 \
        --phi_inertia 0.2 \
        --phi_cognitive 0.2 \
        --phi_social 0.2 \
        --phi_repel 0.1 \
        --cross_rate 0.3 \
        --individual_mutation_rate 0.1 \
        --gene_mutation_rate 0.1 \
        --sigma 0.001 \
        --do_init \
        --elite_percent 0.1 \
        --early_stop_iter 5 \
        --seed $SEED \
        --combine_method $COMBINE_METHOD \
        --cross_method $CROSS_METHOD \
        --plot_enabled \
        --method $METHOD \
        --selection_method $SELECTION_METHOD \
        --ports 9112
    
    echo "Process with seed $SEED completed, wait 10 seconds"
    sleep 10
done