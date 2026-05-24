TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
mkdir -p logs

LR_scalar=${1:-1e-6}
LR_contrast=${2:-5e-7}
batch_size=${3:-8}
gradient_accumulation_steps=${4:-2}
num_epochs=${5:-2}

EXPNAME="consecutive--scalar-data6000-lr$LR_scalar--contrast-data6000-lr$LR_contrast--bs$batch_size-accum$gradient_accumulation_steps-epoch$num_epochs-$TIMESTAMP"

ARGS="train_rm.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_path data/data_train_rm/reward_data-train.jsonl \
    --valid_path data/data_train_rm/reward_data-valid.jsonl \
    --remove_unused_columns False \
    --max_length 2048 \
    --learning_rate $LR_scalar \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_checkpointing \
    --optim paged_adamw_32bit \
    --lr_scheduler_type cosine \
    --bf16 \
    --warmup_ratio 0.03 \
    --num_train_epochs $num_epochs \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy epoch \
    --save_only_model \
    --report_to tensorboard \
    --output_dir models/$EXPNAME-scalar \
    --deepspeed ds_config.json"
run_cmd="torchrun --nproc_per_node 8 $ARGS"
echo $run_cmd
LOG_PATH=logs/${EXPNAME}-scalar
mkdir -p $LOG_PATH
eval ${run_cmd} 2>&1 | tee ${LOG_PATH}/output_${MLP_ROLE_INDEX}.log

ARGS="train_rm_contrastive.py \
    --model_name models/$EXPNAME-scalar/last_checkpoint \
    --train_path data/data_train_rm/reward_data_contrastive-train.jsonl \
    --valid_path data/data_train_rm/reward_data_contrastive-valid.jsonl \
    --remove_unused_columns False \
    --max_length 2048 \
    --learning_rate $LR_contrast \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_checkpointing \
    --optim paged_adamw_32bit \
    --lr_scheduler_type cosine \
    --bf16 \
    --warmup_ratio 0.03 \
    --num_train_epochs $num_epochs \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy epoch \
    --save_only_model \
    --report_to tensorboard \
    --output_dir models/$EXPNAME-contrast \
    --deepspeed ds_config.json"

run_cmd="torchrun --nproc_per_node 8 $ARGS"
echo $run_cmd
LOG_PATH=logs/${EXPNAME}-contrast
mkdir -p $LOG_PATH
eval ${run_cmd} 2>&1 | tee ${LOG_PATH}/output_${MLP_ROLE_INDEX}.log
