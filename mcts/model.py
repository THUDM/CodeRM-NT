from typing import Any, Dict
from openai import OpenAI

import urllib3
urllib3.disable_warnings()

total_post_times = total_input_tokens = total_output_tokens = 0

class Model:
    @property
    def _default_params(self) -> Dict[str, Any]:
        params = {
            "model": self.model,
            "stream": False,
        }
        for attr_name in ['temperature', 'max_tokens']:
            if hasattr(self, attr_name):
                params[attr_name] = getattr(self, attr_name)
        return params

    def __init__(self, model_name, base_url, api_key):
        self.model: str = model_name
        self.max_tokens: int = 4096
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def post(self, request: Any) -> Any:
        global total_post_times, total_input_tokens, total_output_tokens
        retries = 5
        for _ in range(retries):
            response = self.client.chat.completions.create(**request)
            try:
                choice = response.choices[0]
                if choice.finish_reason != 'stop':
                    print(f"Finish reason: {choice.finish_reason}")
                    raise NotImplementedError
                total_post_times += 1
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
                return response
            except:
                print(f"Response content: {response}")
        raise NotImplementedError
    
    def get_usage(self):
        global total_post_times, total_input_tokens, total_output_tokens
        return {
            "call_times": total_post_times,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        }
    
    def set_usage(self, usage):
        global total_post_times, total_input_tokens, total_output_tokens
        total_post_times = usage["call_times"]
        total_input_tokens = usage["input_tokens"]
        total_output_tokens = usage["output_tokens"]
    
    def generate(self, prompt: str, model: str=None):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }
        request.update(self._default_params)
        if model:
            request['model'] = model
        response = self.post(request)
        return response.choices[0].message.content.rstrip()

    def invoke(
        self,
        messages,
        model=None,
        **kwargs: Any,
    ) -> str:
        request = kwargs
        for i in range(len(messages)):
            messages[i] = {key: messages[i][key] for key in ['role', 'content']}
        request["messages"] = messages
        request.update(self._default_params)
        if model:
            request['model'] = model
        response = self.post(request)
        return response.choices[0].message.content.rstrip()

class LocalModel(Model):
    def __init__(self, model_name, base_url="http://localhost:8000/v1"):
        super().__init__(model_name, base_url, api_key='EMPTY')
