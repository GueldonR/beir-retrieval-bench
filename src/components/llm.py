from openai import OpenAI
from openai.types.chat import ChatCompletion

from src.config.settings import VLLM_URL

#############

## todo: fixa systemprompt från zero shot dense retrevial


class QwenInstance: 
    """
    Acts as a wrapper for the qwen instance.    
    """

    def __init__(self):
        self.url = VLLM_URL
        self.api_key="sk-no-key-required"
        self.model_id = "Qwen/Qwen3.5-9B"
        self.client = OpenAI(base_url=self.url, api_key=self.api_key)
        self.max_tokens = 512
    
    def default_chat(self, message, is_thinking: bool = False) -> ChatCompletion:
        resp = self.client.chat.completions.create(
        model=self.model_id,
        messages=[{"role": "user", "content": message}],
        extra_body={"chat_template_kwargs": {"enable_thinking": is_thinking}},
        max_tokens=self.max_tokens,
        )
        msg = resp.choices[0].message
        return msg.content, 
    

class QwenInstanceHyde(QwenInstance): 
    """
    A subclass that executes generation for hyde   

    Option for both reasoning and wihout
    """

    def hyde_prompt(self, query, is_thinking: bool = False):
        """
        Generates a hypotetical message
        """
        
        hyde_prompt_template = f"""You are a content-generation assistant using the HYDE method. 
        Generate a detailed hypothetical answer document based on the following user query: "{query}"""
        
        resp = self.client.chat.completions.create(
        model=self.model_id,
        messages=[{"role": "system", "content": hyde_prompt_template},
                  {"role": "user", "content": query}],
        extra_body={"chat_template_kwargs": {"enable_thinking": is_thinking}},
        max_tokens=self.max_tokens,
        )
        msg = resp.choices[0].message
        return msg.content
    
