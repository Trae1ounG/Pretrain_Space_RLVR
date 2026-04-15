#!/usr/bin/env bash
# ============================================================================
# DSRL (Dual-Space Reinforcement Learning) - Qwen3-4B on Math
# ============================================================================
# Core method: Pretrain-only warmup for N steps, then switch to RL.
#   - Warmup phase (step < pretrain_loss_disable_step):
#       pretrain_only=True, only pretrain loss (no RL gradient)
#   - RL phase (step >= pretrain_loss_disable_step):
#       pretrain disabled -> pure RL
#   - Pretrain uses response-only p(y) unconditional probability
#   - negative_pretrain=True: only negative samples contribute to pretrain loss
# ============================================================================

set -xeuo pipefail
export WANDB_MODE=online
export WANDB_API_KEY="your_wandb_api_key"
project_name='DSRL'

# ========================== Algorithm ==========================
adv_estimator=grpo

# KL penalty (disabled by default)
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# ========================== DSRL Config ==========================
# pretrain_loss_coef defaults to 0.01 in yaml config, no need to set explicitly.
# Pretrain is enabled when pretrain_loss_disable_step is set.

# After this step, pretrain loss is disabled (pretrain_enabled becomes False).
pretrain_loss_disable_step=20

# If True, pretrain_only is forced True for steps < pretrain_loss_disable_step.
pretrain_only_warmup=True

# Which samples contribute to pretrain loss:
#   negative_pretrain=True  + positive_pretrain=False -> only negative samples
#   negative_pretrain=False + positive_pretrain=True  -> only positive samples
#   negative_pretrain=True  + positive_pretrain=True  -> all samples
negative_pretrain=True
positive_pretrain=False

# Which samples contribute to RL loss:
#   positive_rl=True + negative_rl=True -> standard GRPO (all samples)
positive_rl=True
negative_rl=True

# PPO clipping
clip_ratio_low=0.2
clip_ratio_high=0.2

# ========================== Data ==========================
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 8))

# Loss aggregation: "token-mean" or "seq-mean-token-mean"
loss_agg_mode="token-mean"
pretrain_loss_agg_mode="token-mean"
n_resp_per_prompt=8

# ========================== Paths ==========================
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-1}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}"}
validation_data_dir=${validation_data_dir:-"${PWD}/validation_data"}
MODEL_PATH=${MODEL_PATH:-"/path/to/your/Qwen3-4B"}
DATA_PATH=${DATA_PATH:-"${WORKING_DIR}/data"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_PATH}/math_train.parquet"}
TEST_FILE=${TEST_FILE:-["${DATA_PATH}/aime_2024_20times.parquet","${DATA_PATH}/aime_2025_20times.parquet","${DATA_PATH}/amc2023_20times.parquet"]}

# ========================== Sampling ==========================
temperature=1.0
top_p=1.0
top_k=-1  # 0 for HF rollout, -1 for vLLM rollout

# ========================== Performance ==========================
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((1024 * 10))
infer_ppo_max_token_len=$((1024 * 10))
offload=True
gen_tp=2
prompt_template_type="qwen3_no_thinking"

# ========================== Evaluation ==========================
test_freq_early=5
test_freq_late=25
test_freq_threshold=100

# ========================== Experiment Name ==========================
experiment_name="DSRL_warmup${pretrain_loss_disable_step}_Qwen3-4b_math_bsz128_n${n_resp_per_prompt}_resp8k_clip${clip_ratio_high}_lr1e-6_$(date +%Y%m%d_%H%M%S)"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.prompt_template_type=${prompt_template_type} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.pretrain_norm_adv_by_std=False \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.pretrain_loss_disable_step=${pretrain_loss_disable_step} \
    algorithm.pretrain_only_warmup=${pretrain_only_warmup} \
    actor_rollout_ref.actor.negative_pretrain=${negative_pretrain} \
    actor_rollout_ref.actor.positive_pretrain=${positive_pretrain} \
    actor_rollout_ref.actor.positive_rl=${positive_rl} \
    actor_rollout_ref.actor.negative_rl=${negative_rl} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.pretrain_loss_agg_mode=${pretrain_loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq_early} \
    trainer.test_freq_early=${test_freq_early} \
    trainer.test_freq_late=${test_freq_late} \
    trainer.test_freq_threshold=${test_freq_threshold} \
    trainer.save_freq=50 \
    trainer.total_epochs=6 \
    trainer.default_local_dir=${WORKING_DIR}/checkpoints/${project_name}/${experiment_name} \
    trainer.validation_data_dir=${WORKING_DIR}/valid_data/${project_name}/${experiment_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.resume_mode=disable $@
