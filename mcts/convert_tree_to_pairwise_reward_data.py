import os
import re
import json
import random
from tqdm import tqdm
from typing import List
from termcolor import colored
from dataclasses import dataclass, field
from collections import Counter, defaultdict
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

@dataclass
class TreeNode:
    segment: str
    q: float
    n: int
    parent: int = field(default_factory=int)
    children: List[int] = field(default_factory=list)

    def add_child(self, child):
        self.children.append(child)

    def print_tree(self, nodes: List['TreeNode'], level=0):
        print(" " * level * 2 + str(self))
        for child in self.children:
            nodes[child].print_tree(level + 1)
"""
paths: List[
    Tuple[
        path: List[ Tuple[ segment, q, n ] ],
        full response: str,
        critic: List[ Tuple[ List[ tests: str, outputs: str ], feedback: Optional[str] ] ],
    ]
]
"""
def print_tree(tree, node, level=0):
    print(" " * (level * 4) + colored(str((node, tree[node].parent, tree[node].q, tree[node].n, tree[node].q / tree[node].n)), 'red'))
    print(tree[node].segment)
    for child in tree[node].children:
        print_tree(tree, child, level + 1)

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

with open('data_train_rm/reward_data_contrastive.jsonl', 'w') as f:
    for raw_index, sample_old in tqdm(data):
        treenodes: List[TreeNode] = []
        root = TreeNode(*(sample_old[0][0][0]))
        root.q = min(max(root.q, 0), root.n)
        node_to_segment_map = defaultdict(list) # node index -> (path index in sample, segment id in path excluding the first empty segment)
        treenodes.append(root)
        sample = []
        blocks = []
        for path_info in sample_old:
            path, response, critic = path_info
            # BEGIN FILTER
            if not response:
                continue
            block = get_code_blocks(response)['python']
            if len(block) != 1:
                continue
            block = get_function(block[0])[0]
            if not block:
                continue
            sample.append(path_info)
            blocks.append(block)
        for i, path_info in enumerate(sample):
            path, response, critic = path_info
            block = remove_code_comments(blocks[i])
            code = '\n'.join([remove_code_comments(p[0]) for p in path[1:]])
            if block != code:
                continue
            # END FILTER
            ptr = 0
            for j, p in enumerate(path[1:]):
                child_found = False
                for c in treenodes[ptr].children:
                    if p[0] == treenodes[c].segment:
                        node_to_segment_map[c].append((i, j))
                        ptr = c
                        child_found = True
                        break
                if not child_found:
                    node = TreeNode(*p)
                    node.q = min(max(node.q, 0), node.n)
                    node.parent = ptr
                    treenodes[ptr].children.append(len(treenodes))
                    node_to_segment_map[len(treenodes)].append((i, j))
                    ptr = len(treenodes)
                    treenodes.append(node)
        nodes_chosen = [i for i, node in enumerate(treenodes) if i != 0 and (len(node.children) > 1 or len(node.children) == 0)]
        # END BUILD TREE
        for enum_idx, idx1 in enumerate(nodes_chosen[:-1]):
            for idx2 in nodes_chosen[enum_idx + 1:]:
                # BEGIN FILTER SAME PATH
                backtrack_ptr = idx2
                while backtrack_ptr != 0:
                    backtrack_ptr = treenodes[backtrack_ptr].parent
                    if backtrack_ptr == idx1:
                        break
                if backtrack_ptr == idx1:
                    continue
                score1 = treenodes[idx1].q / treenodes[idx1].n
                score2 = treenodes[idx2].q / treenodes[idx2].n
                if score1 == score2:
                    continue
                # END FILTER
                indice_pair = [idx1, idx2]
                segment_pair = [random.choice(node_to_segment_map[idx]) for idx in indice_pair]
                path_pair = [sample[ segment[0] ][0] for segment in segment_pair]
                full_response_pair = [sample[ segment[0] ][1] for segment in segment_pair]
                process_response_pair = []
                for response_idx in range(2):
                    full_response = full_response_pair[response_idx]
                    path = path_pair[response_idx]
                    block = '\n' + get_function(get_code_blocks(full_response)['python'][0])[0]
                    block_cleaned = '\n' + remove_code_comments(block)
                    path_str = ['\n' + remove_code_comments(p[0]) for p in path[1:]]
                    subsequence_index_in_block = find_subsequence_indices(block_cleaned, block) # block_cleaned -> block
                    subsequence_index_in_block.append(subsequence_index_in_block[-1] + 1)
                    block_offset = full_response.find(block)
                    block_cleaned_cutoff_indices = [len(p) for p in path_str]
                    block_cleaned_cutoff_indices = [sum(block_cleaned_cutoff_indices[:i]) for i in range(1, len(block_cleaned_cutoff_indices) + 1)]
                    block_cutoff_indices = [block_offset + subsequence_index_in_block[b] for b in block_cleaned_cutoff_indices] # end at last code
                    block_cutoff_indices[-1] = len(full_response) # end at end of response
                    cutoff = block_cutoff_indices[ segment_pair[response_idx][1] ]
                    process_response_pair.append(full_response[:cutoff])
                win_idx = random.choice(range(2))
                if score1 > score2:
                    win_idx = 0
                elif score1 < score2:
                    win_idx = 1
                f.write(json.dumps({'raw_index': raw_index, 'input': prompts[int(raw_index)], 'output_chosen': process_response_pair[win_idx], 'output_reject': process_response_pair[1 - win_idx], 'q_chosen': treenodes[ indice_pair[win_idx] ].q, 'n_chosen': treenodes[ indice_pair[win_idx] ].n, 'q_reject': treenodes[ indice_pair[1 - win_idx] ].q, 'n_reject': treenodes[ indice_pair[1 - win_idx] ].n}, ensure_ascii=False) + '\n')

data = list(map(json.loads, open('data_train_rm/reward_data_contrastive.jsonl').readlines()))
with open('data_train_rm/reward_data_contrastive-valid.jsonl', 'w') as f:
    for d in data[:round(len(data) * 0.05)]:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
with open('data_train_rm/reward_data_contrastive-train.jsonl', 'w') as f:
    for d in data[round(len(data) * 0.05):]:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
