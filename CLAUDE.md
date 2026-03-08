# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SQL-R1-MORL extends the SQL-R1 paper (NeurIPS 2025) with Multi-Objective Reinforcement Learning. It trains a SQL-R1-3B model to generate SQL from natural language, optimizing two objectives simultaneously — accuracy and efficiency — using PC-Grad gradient projection within the GRPO algorithm. Built on ByteDance's verl RL framework.

## Commands

```bash
# Install (Torch 2.4.0 + CUDA 12.1 + Flash Attention)
bash install.sh

# Train (GRPO with PC-Grad, 2x GPU)
sh sh/train.sh

# Inference (edit MODEL_ENV and DATASET in script first)
sh sh/inference.sh

# Evaluation
sh sh/eval_spider.sh
sh sh/eval_bird.sh

# Run tests (pytest available via optional dep)
pip install -e ".[test]"
pytest

# Clean up stale Ray processes
sh sh/cleanup_ray.sh
```

Training entry point: `python -m verl.trainer.main_ppo` with Hydra config overrides (see `sh/train.sh` for all flags).

## Architecture

### Training Flow

`verl/trainer/main_ppo.py` → creates `RewardManager` + workers → `RayPPOTrainer.fit()`

1. **Rollout**: vLLM generates n=16 SQL responses per prompt
2. **Reward**: `RewardManager` calls `synsql.compute_score()` → returns (accuracy, efficiency) per sample
3. **Advantage**: GRPO normalizes rewards within each prompt group, separately per objective
4. **Actor update**: Two forward passes (one per objective), gradients combined via PC-Grad projection
5. **KL penalty**: Applied once (on accuracy pass only) against frozen reference policy

### Key Files

| File | Role |
|------|------|
| `verl/trainer/main_ppo.py` | Entry point, RewardManager (vector rewards) |
| `verl/workers/actor/dp_actor.py` | Actor with PC-Grad dual-pass training |
| `verl/trainer/ppo/ray_trainer.py` | RayPPOTrainer, per-objective advantage computation |
| `verl/trainer/ppo/core_algos.py` | GRPO advantage normalization, policy loss, KL penalty |
| `verl/utils/reward_score/synsql.py` | Reward function: accuracy (format+exec+match, range [-1.5, 6]) and efficiency (EXPLAIN QUERY PLAN cost, range [0, 1]) |
| `verl/trainer/config/ppo_trainer.yaml` | Default Hydra config (`enable_pc_grad: False` by default) |
| `src/inference.py` | vLLM-based inference for Spider/BIRD benchmarks |
| `src/evaluation_spider.py` | Spider evaluation (exec accuracy) |

### MORL / PC-Grad

- Controlled by `actor_rollout_ref.actor.enable_pc_grad=True` in train config
- `dp_actor.py:pc_grad_combine()` — detects gradient conflicts (negative dot product) and projects conflicting gradients onto the normal plane of each other
- Two separate forward-backward passes per micro-batch avoid `retain_graph` issues with FSDP
- Fast path: skips dual-pass PC-Grad and falls through to single-objective when all efficiency advantages are zero

### Data Protocol

verl uses `DataProto` (defined in `verl/protocol.py`) as a unified tensor/non-tensor container passed between workers. Reward fields `accuracy_rewards` and `efficiency_rewards` are stored as separate batch keys and overwritten with per-objective advantages after GRPO normalization.

## Critical Dependencies

Pinned versions matter — vLLM 0.6.3 requires numpy<2.0, torch==2.4.0, and transformers<4.48. Use `install.sh` for correct wheel resolution.

## Environment Variables (training)

Set in `sh/train.sh`: `VLLM_ATTENTION_BACKEND=XFORMERS`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `NCCL_TIMEOUT=1800`, `RAY_object_store_memory=10737418240`. Optionally set `WANDB_API_KEY` for logging.

## Conventions

- Config via Hydra CLI overrides (dot-notation), defaults in `ppo_trainer.yaml`
- Training data as parquet in `data/`, model weights in `models/`
- Checkpoints and logs written to `logs/`
- Commit messages: short one-liners
- Do NOT commit or push unless the user explicitly says to — always wait for their go-ahead
- Always use `scripts/summarize_wandb.py` to fetch and analyze W&B runs (WandB pages are JS-rendered SPAs that WebFetch cannot scrape)
