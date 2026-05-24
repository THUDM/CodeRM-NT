import os
import re
import json
import yaml
from tqdm import tqdm
import argparse
import logging
import random
from time import time
from collections import defaultdict
from multiprocessing import Process

from model import LocalModel
from mcts_magicoder import MCTS

def worker(id, args, instructions, model_config, lock):
    model = LocalModel(model_config['name'], base_url=model_config['url'])
    all_history = []
    time_start = time()
    token_record_path = os.path.join(args.output_dir, f'token_counts_{id}.json')
    if os.path.exists(token_record_path):
        model_usage = json.load(open(token_record_path))
        model.set_usage(model_usage)
        time_start -= model_usage['elapsed_time']
    for user_input in tqdm(instructions):
        if os.path.exists(os.path.join(args.output_dir, str(user_input['raw_index']) + '.json')):
            continue
        logging.info(f"Start {user_input['raw_index']}")
        tree = MCTS(model, tests=user_input['tests'], use_reflection=True, branching=args.max_branches)
        root = tree.initiate(instruction=user_input["problem"], response=user_input['rewritten_solution'], rewrite=False, lock=lock)
        for _ in range(args.num_rollout - 1):
            tree.do_rollout(root, lock)
        tree_dump = tree.dump()
        json.dump(tree_dump, open(os.path.join(args.output_dir, str(user_input['raw_index']) + '.json'), "w"), indent=2, ensure_ascii=False)
        model_usage = model.get_usage()
        model_usage['elapsed_time'] = time() - time_start
        json.dump(model_usage, open(token_record_path, 'w'), indent=2)
    return all_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_config_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_rollout', type=int, default=10)
    parser.add_argument('--max_branches', type=int, default=-1)
    args = parser.parse_args()
    if args.max_branches == -1:
        args.max_branches = args.num_rollout * 2

    logging.basicConfig(level=logging.INFO)
    logging.info("Initialization")
    instructions = list(map(json.loads, open(args.data_path, "r").readlines()))
    for i in range(len(instructions)):
        instructions[i]['tests'] = [ instructions[i]['test'] ]
    instructions = [i for i in instructions if not os.path.exists(os.path.join(args.output_dir, str(i['raw_index']) + '.json'))]

    logging.info(f"{len(instructions)} remaining instructions")

    os.makedirs(args.output_dir, exist_ok=True)

    model_configs = yaml.safe_load(open(args.model_config_path, "r"))['models']
    num_workers = len(model_configs) * 4

    k, m = divmod(len(instructions), num_workers)
    instructions_workers = [instructions[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_workers)]

    processes = []
    for i in range(num_workers):
        p = Process(target=worker, args=(i, args, instructions_workers[i], model_configs[i % len(model_configs)], None))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    names = sorted([d for d in os.listdir(args.output_dir) if d.split('.')[0].isdigit()])

    token_counts = defaultdict(int)
    for count in [json.load(open(os.path.join(args.output_dir, d))) for d in os.listdir(args.output_dir) if d.startswith('token_counts_') and d.endswith('.json')]:
        for k in count:
            if k != 'elapsed_time':
                token_counts[k] += count[k]
            else:
                token_counts[k] = max(token_counts[k], count[k])
    json.dump(dict(token_counts), open(os.path.join(args.output_dir, 'token_counts.json'), 'w'))
