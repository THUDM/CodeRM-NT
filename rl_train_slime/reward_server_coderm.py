import re
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

reward_model_path = "Rishubi/CodeRM-NT"
device = "cuda:0"
model = AutoModelForSequenceClassification.from_pretrained(reward_model_path, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
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

@app.post("/predict")
def predict(data: InputData):
    print(data)
    question = isolate_user_message(data.question)
    answer = isolate_assistant_message(data.response)
    print(f"Question:\n{question}\nResponse:\n{answer}")
    messages = [{"role": "user", "content": question},
                {"role": "assistant", "content": answer}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        rewards = outputs.logits
        print(rewards)
    rewards = rewards.squeeze(1).float().cpu().numpy().tolist()[0]
    rewards = np.clip(rewards, -4, 4)
    
    return {"rewards": rewards}

# Run the server with:
# CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/root/Megatron-LM uvicorn reward_server:app --host 0.0.0.0 --port 8100
