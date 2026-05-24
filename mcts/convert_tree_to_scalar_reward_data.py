import os
import re
import json
from tqdm import tqdm
from collections import Counter
from astroid import nodes
from astroid.builder import AstroidBuilder
from utils import get_code_str, get_code_blocks, remove_code_comments

def get_function(code, entry=None):
    """ function part, part after the function, entry name """
    tree = AstroidBuilder().string_build(code)
    cutoff, function_name = None, entry
    for ele in tree.body:
        if isinstance(ele, nodes.FunctionDef) or isinstance(ele, nodes.ClassDef):
            cutoff = ele.end_lineno
            if entry is None:
                function_name = ele.name
            if entry is not None and ele.name == entry:
                break
    return ('\n'.join(code.split('\n')[:cutoff]).strip(), '\n'.join(code.split('\n')[cutoff:]).strip(), function_name) if cutoff is not None else (None, None, None)

data_path = 'data_train_rm/tree/treesearch-qwen32b-branch4-10'
items = sorted([item for item in os.listdir(data_path) if item.endswith('.json') and item[:-5].isdigit()], key=lambda x: int(x[:-5]))
data = [(item.split('.')[0], json.load(open(os.path.join(data_path, item)))) for item in tqdm(items)]

prompts = list(map(json.loads, open('data_train_rm/exec_results.jsonl').readlines()))
prompts = {p['raw_index']: p['problem'] for p in prompts}

def find_subsequence_indices(subseq, seq):
    indices = []
    i, j = 0, 0
    while i < len(subseq) and j < len(seq):
        if subseq[i] == seq[j]:
            indices.append(j)
            i += 1
        j += 1
    if i == len(subseq):
        return indices

with open('data_train_rm/reward_data.jsonl', 'w') as f:
    for raw_index, d in tqdm(data):
        for path, full_response, critic in d:
            # BEGIN FILTER
            if not full_response:
                continue
            block = get_code_blocks(full_response)['python']
            if len(block) != 1:
                continue
            block = get_function(block[0])[0]
            if not block:
                continue
            block = remove_code_comments(block)
            code = '\n'.join([remove_code_comments(p[0]) for p in path[1:]])
            if block != code:
                continue
            # END FILTER
            block = '\n' + get_function(get_code_blocks(full_response)['python'][0])[0]
            block_cleaned = '\n' + remove_code_comments(block)
            path_str = ['\n' + remove_code_comments(p[0]) for p in path[1:]]
            subsequence_index_in_block = find_subsequence_indices(block_cleaned, block)
            subsequence_index_in_block.append(subsequence_index_in_block[-1] + 1)
            block_offset = full_response.find(block)
            block_cleaned_cutoff_indices = [len(p) for p in path_str]
            block_cleaned_cutoff_indices = [sum(block_cleaned_cutoff_indices[:i]) for i in range(1, len(block_cleaned_cutoff_indices) + 1)]

            block_cutoff_indices = [block_offset + subsequence_index_in_block[b] for b in block_cleaned_cutoff_indices]
            for i, cutoff in enumerate(block_cutoff_indices):
                f.write(json.dumps({'raw_index': raw_index, 'input': prompts[int(raw_index)], 'output': full_response[:cutoff], 'q': path[i + 1][1], 'n': path[i + 1][2]}, ensure_ascii=False) + '\n')

data = list(map(json.loads, open('data_train_rm/reward_data.jsonl').readlines()))
with open('data_train_rm/reward_data-valid.jsonl', 'w') as f:
    for d in data[:round(len(data) * 0.05)]:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
with open('data_train_rm/reward_data-train.jsonl', 'w') as f:
    for d in data[round(len(data) * 0.05):]:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
