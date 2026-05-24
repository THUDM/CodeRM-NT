#!/bin/bash

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
kill -9 $(ps aux | grep -v grep | grep ray | awk '{print $2}')

set -ex
trap "kill -- -$$" EXIT

REWARD=${1} # rm or test
eps_low=${2:-0.2}
eps_high=${3:-0.28}
num_data=${4:-5000}
rollout_batch_size=${5:-8}
n_samples_per_prompt=${6:-16}

num_rollout=$((num_data / rollout_batch_size))

case "$REWARD" in
    rm)   CUSTOM_RM=code_reward_oci.model_rm ;;
    test) CUSTOM_RM=code_reward_oci.tests_rm ;;
    *) echo "Unknown REWARD=$REWARD" >&2; exit 1 ;;
esac

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXPNAME=qwen3-$REWARD-data$num_data-rbs$rollout_batch_size-numgen$n_samples_per_prompt-epslow$eps_low-epshigh$eps_high-$TIMESTAMP

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B-Thinking.sh"

CKPT_ARGS=(
    --hf-checkpoint /ckpts/Qwen/Qwen3-4B-Thinking-2507
    --ref-load /root/slime/ckpt/qwen3-release/
    --save /root/slime/ckpt/code-rm/$EXPNAME/
    --save-interval $((num_rollout / 50))
)

ROLLOUT_ARGS=(
    --prompt-data /root/slime/data/oci/oci_entry_train.jsonl
    --input-key prompt
    --label-key label
    --metadata-key id
    --apply-chat-template
    --rollout-shuffle

    --custom-rm-path $CUSTOM_RM

    --num-rollout $num_rollout
    --rollout-batch-size $rollout_batch_size
    --n-samples-per-prompt $n_samples_per_prompt
    --rollout-max-response-len 15000
    --rollout-temperature 0.8
    --balance-data
)

EVAL_ARGS=(
    --eval-interval $((num_rollout / 50))
    --eval-prompt-data oci_5k /root/slime/data/oci/oci_entry_val.jsonl
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 15000
    --eval-top-p 0.7
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 4
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --micro-batch-size 1
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip $eps_low
    --eps-clip-high $eps_high
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style cosine
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime-qwen3
    --wandb-group $EXPNAME
    --wandb-key <wandb_key>
    --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
    # default dropout in megatron is 0.1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    # should be good for model performance
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    # need to comment this when using model with MLA
    --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

LOG_PATH=logs-code-rm/${EXPNAME}
mkdir -p $LOG_PATH

RUNTIME_ENV_JSON='{
    "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "'${HAS_NVLINK}'",
        "EXPNAME": "'${EXPNAME}'",
        "PYTHONUNBUFFERED": "1"
    }
}'

run_cmd="ray job submit --address='http://127.0.0.1:8265' \
    --runtime-env-json='$RUNTIME_ENV_JSON' \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]}"

echo $run_cmd
eval ${run_cmd} 2>&1 | tee ${LOG_PATH}/output.log
