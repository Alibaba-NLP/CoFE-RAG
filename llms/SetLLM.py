from typing import Optional, List, Mapping, Any, Dict
import os, time, requests
import logging
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.bridge.pydantic import Field

# gpt-3.5-turbo-0301, gpt-3.5-turbo-16k, gpt-4
DEFAULT_OPENAI_MODEL = 'gpt-3.5-turbo-0301'
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 0
DEFAULT_SEED = 1234
DEFAULT_INPUT_TOKEN = 3000
DEFAULT_MAX_TOKEN = 1000

class SetLLM(CustomLLM):
    model: Optional[str] = Field(
        default=DEFAULT_OPENAI_MODEL, 
        description="The model to use."
    )
    temperature: Optional[float] = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        gte=0.0,
        lte=1.0,
    )
    top_p: Optional[float] = Field(
        default=DEFAULT_TOP_P,
        description="The model considers the results of the tokens with top_p probability mass",
        gte=0.0,
        lte=1.0,
    )
    seed: Optional[int] = Field(
        default=DEFAULT_SEED,
        description="Random seed",
        gt=0,
    )
    max_tokens: Optional[int] = Field(
        default=DEFAULT_MAX_TOKEN,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    n: Optional[int] = Field(
        default=1,
        description="How many completions to generate for each prompt.",
        gt=0,
    )
    max_retries: Optional[int] = Field(
        default=3,
        description="The maximum number of API retries.",
        gte=0,
    )
    max_input_tokens: Optional[int] = Field(
        default=DEFAULT_INPUT_TOKEN,
        description="The maximum number of input tokens.",
        gt=0,
    )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            max_tokens=self.max_tokens,
            n=self.n,
            max_retries=self.max_retries,
        )

    def process(self, query):
        """
        dynamic_max_tokens: usually used with prompt_len: "min(gen_kargs['max_new_tokens'], model_max_length-prompt_len)"
        """
        api_key = 'YOUR_API_KEY'
        # ChatGPT API 的 URL
        url = 'https://api.openai.com/v1/chat/completions'

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "model": self.model,
                "messages": [{"role": "user", "content": query}],
                "n": self.n,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
            }
        try:
            # print(query)
            response = requests.post(url, json=payload, headers=headers)
            res = response.json()
            return True, res['data']['response']['choices'][0]['message']['content']
            
        except:
            return False, query

    def inference(self, query: str) -> str:
        """
        dynamic_max_tokens: usually used with prompt_len: "min(gen_kargs['max_new_tokens'], model_max_length-prompt_len)"
        """
        try_count = 0
        response = None
        while try_count < self.max_retries:
            flag, res = self.process(query)
            if not flag:
                time.sleep(0.5)
                query = res
                try_count += 1
            else:
                response = res
                break
        if response is None:
            logging.error(f"Failed to get response for query: {query}")
        return response

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.inference(prompt)
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = self.inference(prompt)
        for token in response:
            response += token
            yield CompletionResponse(text=response, delta=token)





