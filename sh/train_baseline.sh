# Baseline training: single-objective (accuracy only, no PC-Grad)
# Load WANDB_API_KEY from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_TRIM_THRESHOLD_=0
export RAY_memory_monitor_refresh_ms=0
export RAY_object_store_memory=10737418240
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800

DATA_DIR_PATH=data

# Environment Variables
RUN_ID=7B-baseline
GPU_ENV=2xA100
MODEL_ENV=Qwen2.5-Coder-7B-Instruct
PROJECT_NAME=SQL-R1-MORL

# Paths
LOG_PATH=logs/$PROJECT_NAME
MODEL_PATH=models/$MODEL_ENV
EXPERIMENT_NAME=$GPU_ENV-$MODEL_ENV-$RUN_ID

mkdir -p $LOG_PATH/$MODEL_ENV

# Show sample training and testing data
echo "=== Sample Training Data ==="
python -c "
import pandas as pd
df = pd.read_parquet('$DATA_DIR_PATH/train.parquet')
print(f'Train set: {len(df)} samples, columns: {list(df.columns)}')
for i, row in df.head(2).iterrows():
    prompt = row['prompt'][0]['content'] if isinstance(row['prompt'], list) else str(row['prompt'])
    gt = row.get('reward_model', {})
    if isinstance(gt, str): import ast; gt = ast.literal_eval(gt)
    inner = gt.get('ground_truth', gt) if isinstance(gt, dict) else {}
    db_id = inner.get('db_id', 'N/A') if isinstance(inner, dict) else 'N/A'
    gold_sql = inner.get('sql', 'N/A') if isinstance(inner, dict) else 'N/A'
    print(f'\n--- Train sample {i} ---')
    print(f'DB: {db_id}')
    print(f'Prompt: {prompt[:200]}...')
    print(f'Gold SQL: {gold_sql[:200]}')
"
echo ""
echo "=== Sample Test Data ==="
python -c "
import pandas as pd
df = pd.read_parquet('$DATA_DIR_PATH/test.parquet')
print(f'Test set: {len(df)} samples, columns: {list(df.columns)}')
for i, row in df.head(2).iterrows():
    prompt = row['prompt'][0]['content'] if isinstance(row['prompt'], list) else str(row['prompt'])
    gt = row.get('reward_model', {})
    if isinstance(gt, str): import ast; gt = ast.literal_eval(gt)
    inner = gt.get('ground_truth', gt) if isinstance(gt, dict) else {}
    db_id = inner.get('db_id', 'N/A') if isinstance(inner, dict) else 'N/A'
    gold_sql = inner.get('sql', 'N/A') if isinstance(inner, dict) else 'N/A'
    print(f'\n--- Test sample {i} ---')
    print(f'DB: {db_id}')
    print(f'Prompt: {prompt[:200]}...')
    print(f'Gold SQL: {gold_sql[:200]}')
"
echo ""
echo "========================================="
echo ">>> BASELINE MODE: PC-Grad DISABLED (single-objective accuracy only)"
echo "========================================="

set -x

nvidia-smi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR_PATH/train.parquet \
    data.val_files=$DATA_DIR_PATH/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.enable_pc_grad=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$LOG_PATH/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 $@ 2>&1 | tee $LOG_PATH/$MODEL_ENV/grpo_baseline.log
