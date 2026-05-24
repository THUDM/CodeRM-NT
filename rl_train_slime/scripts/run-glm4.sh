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
num_data=${4:-9000}
rollout_batch_size=${5:-8}
n_samples_per_prompt=${6:-8}

num_rollout=$((num_data / rollout_batch_size))

case "$REWARD" in
    rm)   CUSTOM_RM=code_reward_kodcode.model_rm ;;
    test) CUSTOM_RM=code_reward_kodcode.tests_rm ;;
    *) echo "Unknown REWARD=$REWARD" >&2; exit 1 ;;
esac

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXPNAME=glm4-$REWARD-data$num_data-rbs$rollout_batch_size-numgen$n_samples_per_prompt-epslow$eps_low-epshigh$eps_high-$TIMESTAMP

export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"

CKPT_ARGS=(
    --hf-checkpoint /ckpts/zai-org/GLM-4-9B-0414/
    --ref-load /root/slime/ckpt/glm4-release/
    --save /root/slime/ckpt/code-rm/$EXPNAME/
    --save-interval $((num_rollout / 10))
)

ROLLOUT_ARGS=(
    --prompt-data /root/slime/data/kodcode/kodcode-train.jsonl
    --input-key prompt
    --label-key label
    --metadata-key conversation_id
    --apply-chat-template
    --rollout-shuffle

    --custom-rm-path $CUSTOM_RM

    --num-rollout $num_rollout
    --rollout-batch-size $rollout_batch_size
    --n-samples-per-prompt $n_samples_per_prompt
    --rollout-max-response-len 2048
    --rollout-temperature 0.8
    --balance-data
)

EVAL_ARGS=(
    --eval-interval $((num_rollout / 20))
    --eval-prompt-data kodcode /root/slime/data/kodcode/kodcode-test.jsonl
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 2048
    --eval-top-p 0.7
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 2
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --micro-batch-size 1
    --max-tokens-per-gpu 4608
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
    --eps-clip $eps_low
    --eps-clip-high $eps_high
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime-glm4
    --wandb-group $EXPNAME
    --wandb-key <wandb_key>
    --disable-wandb-random-suffix
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
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
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 6 --disable-usage-stats

LOG_PATH=logs-code-rm/${EXPNAME}
mkdir -p $LOG_PATH

RUNTIME_ENV_JSON='{
    "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "EXPNAME": "'${EXPNAME}'",
        "PYTHONUNBUFFERED": "1"
    }
}'

run_cmd="ray job submit --address='http://127.0.0.1:8265' \
    --runtime-env-json='$RUNTIME_ENV_JSON' \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --rollout-num-gpus 2 \
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
