from dataclasses import dataclass, field
from typing import Optional

import os
import wandb
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import HfArgumentParser, AutoTokenizer
from grpo_config import GRPOConfig
from grpo_trainer import GRPOTrainer
from reward_execution_kodcode import get_execution_results

@dataclass
class Args:
    model_name: Optional[str] = field(default="Qwen/Qwen2.5-Coder-32B-Instruct")
    data_path: Optional[str] = field(default="")
    wandb_key: Optional[str] = field(default="")

if __name__ == '__main__':
    parser = HfArgumentParser((GRPOConfig, Args))
    training_args, args = parser.parse_args_into_dataclasses()
    if training_args.local_rank == 0:
        print(training_args)
        print(args)
    wandb.login(key=args.wandb_key)
    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)

    prompt_template = """Can you solve the following problem with Python code?
{question}
"""
    prompt_template_entrypoint = """Can you solve the following problem with Python code? The function in this code should be named `{test_entry_point}`.
{question}
"""
    def collate(sample):
        if sample['test_entry_point'] is not None:
            return {'prompt': [
                {'role': 'system', 'content': "You are an intelligent programming assistant to produce Python algorithmic solutions"},
                {'role': 'user', 'content': prompt_template_entrypoint.format(question=sample['question'], test_entry_point=sample['test_entry_point'])},
                {'role': 'assistant', 'content': '```python\n'},
            ]}
        return {'prompt': [
            {'role': 'system', 'content': "You are an intelligent programming assistant to produce Python algorithmic solutions"},
            {'role': 'user', 'content': prompt_template.format(question=sample['question'])},
            {'role': 'assistant', 'content': '```python\n'},
        ]}

    data = load_dataset(args.data_path, split='train')
    data = data.filter(lambda x: x['split'] == 'instruct')
    data = data.shuffle(seed=training_args.seed).select(range(9000))
    data = data.map(collate)
    data = data.train_test_split(test_size=0.01, seed=training_args.seed)
    print(data)

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=get_execution_results,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['test'],
    )
    trainer.train()

    print("Saving last checkpoint of the model")
    save_path = os.path.join(training_args.output_dir, "last_checkpoint")
    trainer.save_model(save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(save_path)
