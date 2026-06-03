import os
from typing import Mapping

from openai import OpenAI


class OpenAICompatibleClient:
    """A tiny wrapper around any OpenAI-compatible chat completion API."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        print("Calling large language model...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            answer = response.choices[0].message.content
            print("Large language model responded successfully.")
            return answer or ""
        except Exception as error:
            print(f"Error occurred when calling LLM API: {error}")
            return "Error: Error occurred when calling language model service."


class HelloAgentsLLM:
    """Chapter-style LLM client with a streaming ``think`` method."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ):
        env = env or os.environ
        self.model = model or env.get("LLM_MODEL_ID") or env.get("MODEL_ID") or env.get("MODEL_NAME")
        api_key = api_key or env.get("LLM_API_KEY") or env.get("API_KEY")
        base_url = base_url or env.get("LLM_BASE_URL") or env.get("BASE_URL")
        timeout = timeout or int(env.get("LLM_TIMEOUT", 60))

        if not all([self.model, api_key, base_url]):
            raise ValueError("Model, API key, and base URL must be provided.")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(self, messages: list[dict[str, str]], temperature: float = 0) -> str:
        print(f"Calling {self.model} model...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)
        except Exception as error:
            print(f"Error occurred when calling LLM API: {error}")
            return ""
