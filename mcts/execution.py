import ast
import time
import random
import requests

def syntax_correct_python(code):
    try:
        ast.parse(code)
        return True
    except:
        return False

urls = [
    '<execution_server_url>', # start an execution server with server_execute.py
]

def run_python_with_server(code, urls=urls, timeout=15, max_retries=3, return_exit_status=False, lock=None):
    while True:
        result = requests.post(random.choice(urls), json={'code': code, 'timeout': timeout}).json()
        if 'error' not in result:
            break
        time.sleep(5 + random.random() * 5)
    stdout = result['stdout']
    stderr = result['stderr']
    log = result['log']
    exit_status = {'StatusCode': result['exit_status']}
    if return_exit_status:
        return stdout, stderr, log, exit_status
    else:
        return stdout, stderr, log
