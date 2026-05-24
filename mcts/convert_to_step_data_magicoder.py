import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset
from multiprocessing import Process
from model import LocalModel
from mcts_magicoder import get_response

model = LocalModel('Qwen/Qwen2.5-Coder-32B-Instruct', base_url='http://localhost:8000/v1')

def worker(index, dataset, output_path, start_from):
    processed_items = []
    with open(output_path, "w") as output_file, open(output_path.replace('.jsonl', '_error.txt'), 'w') as f_error:
        for i, item in enumerate(tqdm(dataset, desc=f"Part {index}")):
            try:
                rewritten_response, func, _, _ = get_response(item['problem'], model, item['solution'])
                item = {k: item[k] for k in ['raw_index', 'index', 'problem', 'solution']}
                item['rewritten_solution'] = rewritten_response
                processed_items.append(item)
                output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                output_file.flush()
            except:
                f_error.write(f'{i + start_from}\n')
                f_error.flush()

if __name__ == "__main__":
    parser = ArgumentParser(description="Process a JSONL file using Hugging Face Dataset.")
    parser.add_argument("--input", default='ise-uiuc/Magicoder-OSS-Instruct-75K', help="HuggingFace dataset repo id.")
    parser.add_argument("--output_dir", default='data_train_rm', help="Path to the output JSONL file.")
    parser.add_argument("--num_start", type=int, default=0, help="Number of items to start from.")
    parser.add_argument("--num_items", type=int, required=True, help="Number of items to process.")

    args = parser.parse_args()

    dataset = load_dataset(args.input, split="train").filter(lambda x: x['lang'] == 'python')

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if args.num_items == -1:
        args.num_start = 0
        args.num_items = len(dataset)
    dataset = dataset.select(range(args.num_start, args.num_start + args.num_items))
    output_path = os.path.join(output_dir, f'python-{args.num_start}-{args.num_start + args.num_items}.jsonl')

    num_workers = 8

    k, m = divmod(len(dataset), num_workers)
    dataset_workers = [dataset.select(range(i * k + min(i, m), (i + 1) * k + min(i + 1, m))) for i in range(num_workers)]
    prefix_sum = list(map(len, dataset_workers))
    for i in range(1, len(prefix_sum)):
        prefix_sum[i] += prefix_sum[i - 1]
    prefix_sum = [0] + prefix_sum[:-1]

    processes = []
    for i in range(num_workers):
        print(f"Part {i}: {len(dataset_workers[i])}")
    for i in range(num_workers):
        p = Process(target=worker, args=(i, dataset_workers[i], output_path.replace('.json', f'_{i}.json'), prefix_sum[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    results = [i for j in [list(map(json.loads, open(output_path.replace('.json', f'_{k}.json'), encoding='utf-8').readlines())) for k in range(num_workers)] for i in j]

    with open(output_path, "w") as output_file:
        for i, item in enumerate(tqdm(results)):
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_file.flush()
