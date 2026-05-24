import os
import re
import subprocess
import tempfile
import json
import traceback
import requests
try:
    import pytest
except:
    raise ImportError("Please install pytest to run the code execution tests. Remember to also install pytest-json-report and pytest-timeout!\nUse: pip install pytest pytest-json-report pytest-timeout")

reward_server_ip = "localhost"

def score_by_test(response, test):
    pattern = r'```python\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        return -1
    code = matches[0].strip()
    reward = 0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(test)
            result = subprocess.run(
                ["python", "-m", "pytest", "--json-report", tmpdir],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            json_report_path = os.path.join(tmpdir, ".report.json")
            if not os.path.exists(json_report_path):
                raise ImportError(f"You forgot to install pytest-json-report. Please install it and try again. Remember to also install pytest-json-report and pytest-timeout!\nUse: pip install pytest pytest-json-report pytest-timeout")
            json_report = json.load(open(json_report_path, 'r'))
            result = json_report['summary']
            if 'passed' in result and 'total' in result:
                reward = result['passed'] / result['total']
    except Exception as e:
        print(traceback.format_exc())
        reward = -1
    return reward

def score_by_rm(prompt, response):
    reward = -4.0
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
    response = sample.response
    test = sample.label

    reward = score_by_test(response, test)
    return reward

async def model_rm(args, sample, **kwargs):
    prompt = sample.prompt
    response = sample.response

    reward = score_by_rm(prompt, response)
    return reward
