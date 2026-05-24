import os
import json
import random
from tqdm import tqdm
from model import LocalModel
from mcts_magicoder import get_function, get_code_str, get_test
from execution import run_python_with_server
from multiprocessing import Process

urls = [
    'http://localhost:5000', # start an execution server with server_execute.py
]

def worker(id, data_worker, output_path):
    model = LocalModel('Qwen/Qwen2.5-Coder-32B-Instruct', base_url=f"http://localhost:8000/v1")
    with open(output_path, 'w') as f:
        for d in tqdm(data_worker, desc=f'Worker {id}'):
            if d.get('rewritten_solution', None) is None:
                continue
            code, test, funcname = get_function(get_code_str(d['rewritten_solution'])['python'])
            if code is None:
                continue
            print(d['raw_index'])
            if not test:
                test = get_test(d['problem'], d['rewritten_solution'], model)
            code = code.rstrip() + '\n\n' + test
            stdout, stderr, log = run_python_with_server(code, urls, timeout=15)
            # Lengthy output
            len_threshold = 5000
            if len(stdout) > len_threshold:
                stdout = stdout[:len_threshold // 2] + ' ... ' + stdout[-len_threshold // 2:]
            if len(stderr) > len_threshold:
                stderr = stderr[:len_threshold // 2] + ' ... ' + stderr[-len_threshold // 2:]
            if len(log) > len_threshold:
                log = log[:len_threshold // 2] + ' ... ' + log[-len_threshold // 2:]
            d['test'] = test
            d['execution_result'] = (stdout, stderr, log)
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            f.flush()

if __name__ == '__main__':
    output_path = 'data_train_rm/exec_results.jsonl'
    num_workers = 8
    data = list(map(json.loads, open('data_train_rm/python-0-12000.jsonl').readlines()))
    for i in range(num_workers):
        if os.path.exists(output_path.replace('.jsonl', f'_{i}.jsonl')):
            with (open(output_path, 'a') if os.path.exists(output_path) else open(output_path, 'w')) as f:
                f.write(open(output_path.replace('.jsonl', f'_{i}.jsonl')).read())
            os.remove(output_path.replace('.jsonl', f'_{i}.jsonl'))
    if os.path.exists(output_path):
        data_exist = [json.loads(line) for line in open(output_path).readlines()]
        id_exist = set()
        with open(output_path, 'w') as f:
            for d in data_exist:
                if d['raw_index'] not in id_exist:
                    id_exist.add(d['raw_index'])
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
        id_exist = set([json.loads(line)['raw_index'] for line in open(output_path).readlines()])
        data = [d for d in data if d['raw_index'] not in id_exist]
    data = [d for d in tqdm(data) if d.get('rewritten_solution', None) and get_function(get_code_str(d['rewritten_solution'])['python'])[0]]
    random.shuffle(data)

    print(f'Processing {len(data)} tasks')
    k, m = divmod(len(data), num_workers)
    data_workers = [data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_workers)]
    processes = []
    for i in range(num_workers):
        p = Process(target=worker, args=(i, data_workers[i], output_path.replace('.jsonl', f'_{i}.jsonl')))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    with (open(output_path, 'a') if os.path.exists(output_path) else open(output_path, 'w')) as f:
        for i in range(num_workers):
            for d in open(output_path.replace('.jsonl', f'_{i}.jsonl')).readlines():
                f.write(d.rstrip() + '\n')
