from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

@dataclass
class Args:
    
    train_path: Union[str, List[str]] = field(default="")
    valid_path: Union[str, List[str]] = field(default="")
    max_length: Optional[int] = field(default=8192)

def build_dataset(tokenizer, data_path):
    def collate(sample):
        sample['output_chosen'] = tokenizer.apply_chat_template([
            {'role': 'user', 'content': sample['input']},
            {'role': 'assistant', 'content': sample['output_chosen']}
        ], add_generation_prompt=False, tokenize=False)
        sample['output_reject'] = tokenizer.apply_chat_template([
            {'role': 'user', 'content': sample['input']},
            {'role': 'assistant', 'content': sample['output_reject']}
        ], add_generation_prompt=False, tokenize=False)
        return sample
    dataset = load_dataset('json', data_files={'train': data_path}, split='train')
    dataset = dataset.map(collate, num_proc=8)
    return dataset

@dataclass
class Collator:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = 'max_length'
    max_length: Optional[int] = None

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer(
            [sample['output_chosen'] for sample in samples] + [sample['output_reject'] for sample in samples],
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt',
            truncation=True
        )
        batch = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'return_loss': True
        }
        return batch

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        score_chosen = outputs.logits[..., :outputs.logits.shape[-2] // 2, :]
        score_reject = outputs.logits[..., outputs.logits.shape[-2] // 2:, :]
        loss = -F.logsigmoid(score_chosen - score_reject).mean()
        print(loss)
        return (loss, {'loss': loss}) if return_outputs else loss

if __name__ == '__main__':
    parser = HfArgumentParser((TrainingArguments, Args))
    training_args, args = parser.parse_args_into_dataclasses()
    training_args.label_names = []
    if training_args.local_rank == 0:
        print(training_args)
        print(args)
    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = build_dataset(tokenizer, args.train_path).shuffle(seed=training_args.seed)
    eval_dataset = build_dataset(tokenizer, args.valid_path).shuffle(seed=training_args.seed)
    if training_args.local_rank == 0:
        print(f"Training set: {train_dataset}, Eval set: {eval_dataset}")
        print(train_dataset[0])
        print(eval_dataset[0])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1, torch_dtype=torch.bfloat16)
    if training_args.local_rank == 0:
        print(model)
    model.config.use_cache = not training_args.gradient_checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(tokenizer=tokenizer, max_length=args.max_length),
    )
    trainer.train()

    print("Saving last checkpoint of the model")
    save_path = os.path.join(training_args.output_dir, "last_checkpoint")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
