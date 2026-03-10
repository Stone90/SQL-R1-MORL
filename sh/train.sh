# SQL-R1-MORL Training (PC-Grad enabled)
# Load WANDB_API_KEY from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Auto-activate venv if present
VENV_PATH="${VENV_PATH:-$(cd "$(dirname "$0")/.." && pwd)/.venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    . "$VENV_PATH/bin/activate"
fi

# Clean up stale Ray processes from previous runs
ray stop --force 2>/dev/null
pkill -9 -f "ray::" 2>/dev/null
pkill -9 -f "gcs_server" 2>/dev/null
rm -rf /tmp/ray 2>/dev/null

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_TRIM_THRESHOLD_=0
export RAY_memory_monitor_refresh_ms=0
export RAY_object_store_memory=4294967296
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_DEBUG=INFO
export RAY_RUNTIME_ENV_AGENT_STARTUP_TIMEOUT_S=120
export RAY_DISABLE_DOCKER_CPU_WARNING=1

DATA_DIR_PATH=data
SYNSQL_DB_DIR=${SYNSQL_DB_DIR:-databases}
export SYNSQL_DB_DIR

# Environment Variables
RUN_ID=3B
GPU_ENV=2xH100
MODEL_ENV=SQL-R1-3B
PROJECT_NAME=SQL-R1-MORL
NUM_CASES=${NUM_CASES:-6400}        # number of training cases to run (default: 6400 = 1 epoch)
TRAIN_BATCH_SIZE=16

# Paths
LOG_PATH=logs/$PROJECT_NAME
MODEL_PATH=models/$MODEL_ENV
EXPERIMENT_NAME=$GPU_ENV-$MODEL_ENV-$RUN_ID

# Colors
BOLD='\033[1m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RED='\033[1;31m'
RESET='\033[0m'

echo ""
echo "${CYAN}╔══════════════════════════════════════════════════════╗${RESET}"
echo "${CYAN}║${RESET}  ${BOLD}SQL-R1-MORL Training (PC-Grad MORL)${RESET}"
echo "${CYAN}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

mkdir -p $LOG_PATH/$MODEL_ENV

# ── Pre-flight checks ──
echo "${GREEN}>>>${RESET} ${BOLD}Pre-flight checks${RESET}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "    ${RED}✗${RESET} Model not found at $MODEL_PATH"
    echo "    ${YELLOW}→${RESET} Run: sh sh/setup_data.sh"
    exit 1
fi
echo "    ${GREEN}✓${RESET} Model: $MODEL_PATH"

if [ ! -f "$DATA_DIR_PATH/train.parquet" ] || [ ! -f "$DATA_DIR_PATH/test.parquet" ]; then
    echo "    ${RED}✗${RESET} Data not found at $DATA_DIR_PATH/"
    echo "    ${YELLOW}→${RESET} Run: sh sh/setup_data.sh"
    exit 1
fi
echo "    ${GREEN}✓${RESET} Data: $DATA_DIR_PATH/train.parquet, test.parquet"

DB_COUNT=$(find -L "$SYNSQL_DB_DIR" -name "*.sqlite" 2>/dev/null | wc -l)
if [ "$DB_COUNT" -eq 0 ]; then
    echo "    ${RED}✗${RESET} No databases found at $SYNSQL_DB_DIR/"
    echo "    ${YELLOW}→${RESET} Run: sh sh/setup_data.sh"
    exit 1
fi
echo "    ${GREEN}✓${RESET} Databases: $DB_COUNT SQLite files in $SYNSQL_DB_DIR/"

if [ -n "$WANDB_API_KEY" ]; then
    echo "    ${GREEN}✓${RESET} WANDB_API_KEY loaded from .env"
else
    echo "    ${YELLOW}→${RESET} WANDB_API_KEY not set — logging to wandb will fail"
fi

echo "    ${GREEN}✓${RESET} Experiment: $EXPERIMENT_NAME"
echo "    ${GREEN}✓${RESET} Logs: $LOG_PATH/$MODEL_ENV/grpo.log"
echo ""

# ── Show GPU info ──
echo "${GREEN}>>>${RESET} ${BOLD}GPU Info${RESET}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
    echo "    ${GREEN}✓${RESET} GPU $line"
done
echo "    ${GREEN}✓${RESET} System RAM:"
free -h | head -2 | while read line; do echo "      $line"; done
echo ""

# ── Show sample data ──
echo "${GREEN}>>>${RESET} ${BOLD}Sample Training Data${RESET}"
python -c "
import pandas as pd, ast

BOLD, GREEN, YELLOW, CYAN, RESET = '\033[1m', '\033[1;32m', '\033[1;33m', '\033[1;36m', '\033[0m'

for split, path in [('Train', '$DATA_DIR_PATH/train.parquet'), ('Test', '$DATA_DIR_PATH/test.parquet')]:
    df = pd.read_parquet(path)
    print(f'    {YELLOW}→{RESET} {BOLD}{split} set:{RESET} {len(df):,} samples')
    for i, row in df.head(2).iterrows():
        prompt = row['prompt'][0]['content'] if isinstance(row['prompt'], list) else str(row['prompt'])
        gt = row.get('reward_model', {})
        if isinstance(gt, str): gt = ast.literal_eval(gt)
        inner = gt.get('ground_truth', gt) if isinstance(gt, dict) else {}
        db_id = inner.get('db_id', 'N/A') if isinstance(inner, dict) else 'N/A'
        gold_sql = inner.get('sql', 'N/A') if isinstance(inner, dict) else 'N/A'
        print(f'    {CYAN}┌─ {split} sample {i}{RESET}')
        print(f'    {CYAN}│{RESET} DB:       {db_id}')
        print(f'    {CYAN}│{RESET} Prompt:   {prompt[:150]}...')
        print(f'    {CYAN}└─{RESET} Gold SQL: {gold_sql[:150]}')
    print()
"

# ── Compute training steps ──
TOTAL_TRAINING_STEPS=$(( NUM_CASES / TRAIN_BATCH_SIZE ))
if [ $TOTAL_TRAINING_STEPS -lt 1 ]; then
    TOTAL_TRAINING_STEPS=1
fi
MAX_STEPS=${MAX_STEPS:-200}
if [ $TOTAL_TRAINING_STEPS -gt $MAX_STEPS ]; then
    TOTAL_TRAINING_STEPS=$MAX_STEPS
fi

# ── Training config summary ──
echo "${GREEN}>>>${RESET} ${BOLD}Training Config${RESET}"
echo "    ${YELLOW}→${RESET} Algorithm:    GRPO + PC-Grad (MORL)"
echo "    ${YELLOW}→${RESET} Cases:        $NUM_CASES → $TOTAL_TRAINING_STEPS steps"
echo "    ${YELLOW}→${RESET} Batch size:   $TRAIN_BATCH_SIZE (mini=4, micro=2)"
echo "    ${YELLOW}→${RESET} Learning rate: 5e-7"
echo "    ${YELLOW}→${RESET} KL loss:      low_var_kl (coef=0.01)"
echo "    ${YELLOW}→${RESET} Entropy coef: 0.01"
echo "    ${YELLOW}→${RESET} Rollout:      n=16, temp=0.7, vLLM (gpu_mem=0.85)"
echo "    ${YELLOW}→${RESET} Dynamic bsz:  max_token_len=24576/gpu"
echo "    ${YELLOW}→${RESET} FSDP:         size=2, grad_ckpt=True, optim_offload=True"
echo "    ${YELLOW}→${RESET} Save/test:    every 50 steps"
echo ""

echo "${CYAN}╔══════════════════════════════════════════════════════╗${RESET}"
echo "${CYAN}║${RESET}  ${BOLD}Starting Training...${RESET}"
echo "${CYAN}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

set -x

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR_PATH/train.parquet \
    data.val_files=$DATA_DIR_PATH/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.enable_pc_grad=True \
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$LOG_PATH/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.max_ckpt_to_keep=5 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=1 $@ 2>&1 | tee $LOG_PATH/$MODEL_ENV/grpo.log
