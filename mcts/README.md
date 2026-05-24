# Reward Data Curation

Implements the MCTS pipeline that produces the training data for CodeRM-NT.

## Quick Start

### 1. Start the LLM server

Both step-data rewriting (step 2) and MCTS (step 5) require a local OpenAI-compatible LLM server. We use Qwen2.5-Coder-32B-Instruct:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2
```

Create a `config_models.yaml` file listing one or more OpenAI-compatible endpoints:

```yaml
models:
  - name: "Qwen/Qwen2.5-Coder-32B-Instruct"
    url: "http://localhost:8000/v1"
  # Add more entries to fan out across additional servers, e.g.:
  # - name: "Qwen/Qwen2.5-Coder-32B-Instruct"
  #   url: "http://192.168.0.2:8000/v1"
```

### 2. Generate step-format source data

Rewrite Python problems from [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) into the step-by-step format for MCTS:

```bash
python convert_to_step_data_magicoder.py --num_items 12000
```

This produces `data_train_rm/python-0-12000.jsonl`.

### 3. Start the code execution server

The execution server runs code inside Docker containers:

```bash
docker pull python:3.11
python server_execute.py --port 5000 --num_workers 4
```

### 4. Gather execution results

Run:

```bash
python gather_execution_result.py
```

This produces `data_train_rm/exec_results.jsonl`.

### 5. Run MCTS

```bash
# args: max_branches num_rollout model_config
bash run_treesearch_magicoder.sh 4 10 config_models.yaml
```

This produces search tree JSON files under `data_train_rm/tree/treesearch-qwen32b-branch{max_branches}-{num_rollout}/`.

### 6. Convert trees to training data

```bash
# Scalar data for MSE training
python convert_tree_to_scalar_reward_data.py

# Pairwise data for contrastive training
python convert_tree_to_pairwise_reward_data.py
```

Output files:

- scalar data: `data_train_rm/reward_data-{train,valid}.jsonl`
- pairwise data: `data_train_rm/reward_data_contrastive-{train,valid}.jsonl`
