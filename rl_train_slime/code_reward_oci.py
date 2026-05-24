import re
import traceback
import requests

reward_server_ip = "localhost"

safe_prelude = """
import os, shutil, subprocess, signal, builtins

def _no_op(*args, **kwargs):
    return None

builtins.exit = _no_op
builtins.quit = _no_op

os.remove = _no_op
os.rmdir = _no_op
os.kill = _no_op
os.system = _no_op
os.putenv = _no_op
os.remove = _no_op
os.removedirs = _no_op
os.rmdir = _no_op
os.fchdir = _no_op
os.setuid = _no_op
os.fork = _no_op
os.forkpty = _no_op
os.killpg = _no_op
os.rename = _no_op
os.renames = _no_op
os.truncate = _no_op
os.replace = _no_op
os.unlink = _no_op
os.fchmod = _no_op
os.fchown = _no_op
os.chmod = _no_op
os.chown = _no_op
os.chroot = _no_op
os.fchdir = _no_op
os.lchflags = _no_op
os.lchmod = _no_op
os.lchown = _no_op
os.getcwd = _no_op
os.chdir = _no_op

shutil.rmtree = _no_op
shutil.move = _no_op
shutil.chown = _no_op

subprocess.Popen = _no_op

if hasattr(signal, "SIGKILL"):
    signal.SIGKILL = None
"""

def score_by_test(prompt, response, test):
    end_pos = response.find('</think>')
    if end_pos == -1:
        return -1
    pattern = r'```python\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        return -1
    code = safe_prelude + '\n' + matches[0].strip() + '\n' + test
    try:
        url = f"http://{reward_server_ip}:8200/execute"
        payload = {
            "question": prompt,
            "response": response,
            "code": code,
        }
        response = requests.post(url, json=payload)
        reward = response.json().get("rewards", -1.0)
    except Exception as e:
        print(traceback.format_exc())
    return reward

def score_by_rm(prompt, response):
    reward = -4.0
    end_pos = response.find('</think>')
    if end_pos == -1:
        return reward
    try:
        url = f"http://{reward_server_ip}:8100/predict"
        payload = {
            "question": prompt,
            "response": response
        }
        response = requests.post(url, json=payload)
        reward = response.json().get("rewards", -4.0)
    except Exception as e:
        print(traceback.format_exc())
    return reward

async def tests_rm(args, sample, **kwargs):
    prompt = sample.prompt
    response = sample.response
    test = sample.label

    reward = score_by_test(prompt, response, test)
    return reward

async def model_rm(args, sample, **kwargs):
    prompt = sample.prompt
    response = sample.response

    reward = score_by_rm(prompt, response)
    return reward
