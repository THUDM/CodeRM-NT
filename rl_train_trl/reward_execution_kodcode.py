import os
import re
import subprocess
import concurrent.futures
import tempfile

pattern = r"```python(.*?)```"

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

def get_execution_results(prompts, completions, test_code, **reward_kwargs):
    prompts = [p[0]['content'] for p in prompts]
    completions = [c[0]['content'] for c in completions]
    codes = [None] * len(prompts)
    for i, completion in enumerate(completions):
        matches = re.findall(pattern, completion, re.DOTALL)
        if matches:
            completion = matches[0].strip()
        codes[i] = completion

    def execute_code(code, test):
        if code is None:
            return -1
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                    f.write(safe_prelude + code)
                with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                    f.write(test)
                result = subprocess.run(
                    ["python", "-m", "pytest", tmpdir],
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )
                return float(1 if result.returncode == 0 else 0)
        except:
            return 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(execute_code, codes, test_code))
    return results
