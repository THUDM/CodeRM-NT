IMAGE=python:3.11
CONTAINERS=$(docker ps -a -q --filter "ancestor=$IMAGE")
if [ -n "$CONTAINERS" ]; then
    docker stop $CONTAINERS
    docker rm $CONTAINERS
    echo "Stopped and removed all containers using the image $IMAGE."
else
    echo "No containers found using the image $IMAGE."
fi

MAX_BRANCHES=${1:-4}
NUM_ROLLOUT=${2:-10}
MODEL_CONFIG=${3:-"config_models.yaml"}
EXPNAME=treesearch-qwen32b-branch$MAX_BRANCHES-$NUM_ROLLOUT
if [ "$MAX_BRANCHES" == "-1" ]; then
    EXPNAME="${EXPNAME//branch-1/unlimitbranching}"
fi

python treesearch_magicoder.py \
    --data_path data_train_rm/exec_results.jsonl \
    --model_config_path $MODEL_CONFIG \
    --output_dir data_train_rm/tree/${EXPNAME} \
    --num_rollout $NUM_ROLLOUT \
    --max_branches $MAX_BRANCHES
