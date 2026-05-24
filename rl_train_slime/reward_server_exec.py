import re
import sys
import tempfile
import subprocess
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

def isolate_user_message(question):
    if '<|user|>' in question and '<|assistant|>' in question:
        re_user = r'<\|user\|>\s*([\d\D]*?)\s*<\|assistant\|>'
        user_message = re.findall(re_user, question)
        if user_message:
            return user_message[0]
    elif '<|im_start|>' in question and '<|im_end|>' in question:
        re_user = r'<\|im_start\|>user\s*([\d\D]*?)\s*<\|im_end\|>'
        user_message = re.findall(re_user, question)
        if user_message:
            return user_message[0]
    return ''

def isolate_assistant_message(response):
    end_pos = response.find('<|user|>')
    if end_pos != -1:
        response = response[:end_pos]
    think_end_token = '</think>'
    think_end_pos = response.rfind(think_end_token)
    if think_end_pos != -1:
        response = response[think_end_pos + len(think_end_token):]
    return response.strip()

class InputData(BaseModel):
    question: str  # Expecting list of floats/numbers
    response: str
    code: str

@app.post("/execute")
def predict(data: InputData):
    print(data)
    question = isolate_user_message(data.question)
    answer = isolate_assistant_message(data.response)
    code = data.code
    print(f"Question:\n{question}\nResponse:\n{answer}\nCode:\n{code}")
    rewards = 0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
        if result.returncode == 0:
            rewards = 1
    except Exception as e:
        pass
    print(rewards)
    return {"rewards": rewards}

# Run the server with:
# uvicorn reward_server_exec:app --host 0.0.0.0 --port 8200
