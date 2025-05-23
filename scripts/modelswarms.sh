export TOKENIZERS_PARALLELISM=true

N=10
ITERS=10
ROOT_DIR="input your root dir here"
MODEL_NAME="input your model name here"

TASKS='math'
TEST_TASKS='math'

WEIGHTS='1.0'
COMBINE_METHOD=linear

SEEDS=(41 42 47 53 3407)

for SEED in "${SEEDS[@]}"; do
    echo "Running PSO with seed: $SEED"
    python run_pso.py \
        --tasks $TASKS \
        --test_tasks $TEST_TASKS \
        --task_weights $WEIGHTS \
        --model_path /$ROOT_DIR/models/$MODEL_NAME \
        --lora_dir /$ROOT_DIR/lora/$MODEL_NAME/lora \
        --population_size $N \
        --max_iter $ITERS \
        --lambda_step 0.1 \
        --phi_lambda 0.95 \
        --phi_inertia 0.2 \
        --phi_cognitive 0.2 \
        --phi_social 0.2 \
        --phi_repel 0.1 \
        --early_stop_iter 5 \
        --plot_enabled \
        --seed $SEED \
        --combine_method $COMBINE_METHOD \
        --ports 9112 36048 22246 13732

    echo "Process with seed $SEED completed, wait 10 seconds"
    sleep 10
done