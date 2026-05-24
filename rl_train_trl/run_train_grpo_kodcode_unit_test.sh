TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
mkdir -p logs
export WANDB_PROJECT=code_rm_kodcode

eps_low=${1:-0.2}
eps_high=${2:-0.2}
LR=${3:-2e-6}
batch_size=${4:-4}
gradient_accumulation_steps=${5:-8}
num_epochs=${6:-1}
num_generations=${7:-7}
temperature=${8:-1.5}
lr_scheduler_type=${9:-linear}
seed=${10:-42}

MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct
EXPNAME=qwen7b-kodcode_instruct_9k-unit_test-lr$LR$lr_scheduler_type-bs$batch_size-accum$gradient_accumulation_steps-epslow$eps_low-high$eps_high-ngen$num_generations-t$temperature-epoch$num_epochs-$TIMESTAMP

output_dir=runs/$EXPNAME
LOG_PATH=logs/${EXPNAME}
mkdir -p $LOG_PATH

ARGS="train_grpo_kodcode_unit_test.py \
    --model_name $MODEL_PATH \
    --data_path KodCode/KodCode-V1 \
    --num_generations $num_generations \
    --temperature $temperature \
    --learning_rate $LR \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_checkpointing \
    --log_completions \
    --max_completion_length 2048 \
    --optim paged_adamw_32bit \
    --lr_scheduler_type $lr_scheduler_type \
    --bf16 1 \
    --warmup_ratio 0.03 \
    --epsilon_low $eps_low \
    --epsilon_high $eps_high \
    --num_train_epochs $num_epochs \
    --seed $seed \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 0.1 \
    --save_only_model \
    --report_to wandb \
    --wandb_key <wandb_key> \
    --output_dir $output_dir \
    --use_vllm \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.5 \
    --deepspeed ds_config_zero3.json"

run_cmd="torchrun --nproc_per_node 7 $ARGS"
echo $run_cmd
eval ${run_cmd} 2>&1 | tee ${LOG_PATH}/output_${MLP_ROLE_INDEX}.log
